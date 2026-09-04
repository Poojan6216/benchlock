"""Configuration: Pydantic v2 models over `benchlock.yaml`, with line-numbered errors.

Two things make this module more than boilerplate:

1. **Line numbers.** A validation failure must name the field *and the line* it is on.
   We compose the YAML into a node tree first, record a path -> line map, then map
   Pydantic's ``loc`` tuples back onto it.
2. **Loud failure (Hard Rule 10).** ``extra="forbid"`` everywhere, so a typo is an error
   rather than a silently ignored key that leaves a default in place.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from benchlock.model.pins import AnchorMode

#: Bumped when the YAML schema changes shape. A config declaring another version is
#: rejected rather than best-effort parsed.
SCHEMA_VERSION = 1

DEFAULT_CONFIG_NAME = "benchlock.yaml"


# --------------------------------------------------------------------------------------
# YAML loading with source positions
# --------------------------------------------------------------------------------------

Loc = tuple[str | int, ...]


def _walk(node: yaml.Node, path: Loc, out: dict[Loc, int]) -> None:
    """Record 1-based source lines for every path in a composed YAML node tree."""
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            key = key_node.value
            child = (*path, key)
            # Point at the key, which is what the user needs to find and edit.
            out[child] = key_node.start_mark.line + 1
            _walk(value_node, child, out)
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            child = (*path, index)
            out[child] = item.start_mark.line + 1
            _walk(item, child, out)


def load_yaml_with_lines(text: str, source: str) -> tuple[dict[str, Any], dict[Loc, int]]:
    """Parse YAML, returning the data and a map from field path to 1-based line number."""
    try:
        node = yaml.compose(text)
        data = yaml.safe_load(text)
    except yaml.MarkedYAMLError as exc:
        line = (exc.problem_mark.line + 1) if exc.problem_mark else 0
        raise ConfigError(
            source,
            [
                ConfigIssue(
                    loc=(),
                    line=line,
                    message=f"the file is not valid YAML: {exc.problem or exc.context}",
                    hint="fix the YAML syntax; benchlock could not parse the file at all",
                )
            ],
        ) from exc

    if data is None:
        raise ConfigError(
            source,
            [
                ConfigIssue(
                    loc=(),
                    line=0,
                    message="the config file is empty",
                    hint="run `benchlock init` to write a starter benchlock.yaml",
                )
            ],
        )
    if not isinstance(data, dict):
        raise ConfigError(
            source,
            [
                ConfigIssue(
                    loc=(),
                    line=1,
                    message="the top level of the config must be a mapping, "
                    f"got {type(data).__name__}",
                    hint="the file should start with keys like `version:` and `alpha:`",
                )
            ],
        )

    lines: dict[Loc, int] = {}
    if node is not None:
        _walk(node, (), lines)
    return data, lines


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """One problem with the config, addressed to the person who has to fix it."""

    loc: Loc
    line: int
    message: str
    hint: str

    @property
    def field(self) -> str:
        return ".".join(str(p) for p in self.loc) if self.loc else "<file>"


class ConfigError(Exception):
    """Raised for any invalid config. Carries every issue, not just the first."""

    def __init__(self, source: str, issues: list[ConfigIssue]) -> None:
        self.source = source
        self.issues = issues
        super().__init__(self.render())

    def render(self) -> str:
        head = f"{len(self.issues)} problem(s) in {self.source}:"
        body = []
        for issue in self.issues:
            where = f"{self.source}:{issue.line}" if issue.line else self.source
            body.append(f"  {where}: [{issue.field}] {issue.message}\n    fix: {issue.hint}")
        return "\n".join([head, *body])


#: Field-specific repair advice. Generic pydantic messages are useless to a user staring
#: at a YAML file at 2am; these name the command or the value that fixes it.
_HINTS: dict[Loc, str] = {
    ("version",): f"set `version: {SCHEMA_VERSION}` — the only schema this build understands",
    ("alpha",): "alpha is the false-alarm budget; use a probability in (0, 0.5], e.g. 0.05",
    ("score_scale",): "declare the rubric's range as two numbers, e.g. `score_scale: [1, 5]`",
    ("min_runs",): "min_runs is how many runs must exist before a verdict is attempted (>= 2)",
    ("min_obs",): "min_obs is the minimum judged items per run (>= 1)",
    ("system", "adapter"): "one of: jsonl, promptfoo, inspect_ai, deepeval",
    ("system", "path"): "path to your eval framework's output file or directory",
    ("anchor", "mode"): "one of: frozen-self (default, no labels), human, replicate",
    ("anchor", "n"): "anchor set size; run `benchlock plan --target-shift 0.05` to size it",
    ("anchor", "cadence"): "re-score anchors every N runs; 1 means every run",
    ("anchor", "selection"): "one of: stratified (recommended), random",
    ("anchor", "noise_replicates"): "K replicates for the noise floor; >= 2, default 5",
    ("judge", "provider"): "one of: anthropic, openai",
    ("judge", "model"): "the exact dated snapshot string, e.g. claude-sonnet-4-5-20250929",
    ("judge", "rubric"): "path to the rubric/system prompt file used by your judge",
    ("gate", "fail_on"): "verdicts that fail CI; default [system, both]. `judge` must not fail CI",
    ("gate", "warn_on"): "verdicts that warn but do not fail; default [indeterminate]",
}


def _hint_for(loc: Loc) -> str:
    for depth in range(len(loc), 0, -1):
        hint = _HINTS.get(loc[:depth])
        if hint:
            return hint
    return "see benchlock.yaml.example for the expected shape"


def _line_for(loc: Loc, lines: dict[Loc, int]) -> int:
    """Best available line: the field itself, else its nearest declared ancestor."""
    for depth in range(len(loc), 0, -1):
        line = lines.get(loc[:depth])
        if line:
            return line
    return 0


def _humanise(err: dict[str, Any]) -> str:
    """Turn a pydantic error dict into a sentence that names the offending value."""
    kind = err.get("type", "")
    msg = str(err.get("msg", "invalid value"))
    if kind == "missing":
        return "required field is missing"
    if kind == "extra_forbidden":
        return (
            "unknown field — check the spelling; benchlock rejects unknown keys "
            "rather than ignoring them"
        )
    given = err.get("input")
    if kind.startswith(("int_", "float_", "string_", "bool_", "list_", "dict_")) or kind in {
        "enum",
        "literal_error",
    }:
        return f"{msg} (got {given!r})"
    return f"{msg} (got {given!r})" if given is not None else msg


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class AdapterKind(StrEnum):
    JSONL = "jsonl"
    PROMPTFOO = "promptfoo"
    INSPECT_AI = "inspect_ai"
    DEEPEVAL = "deepeval"


class SelectionKind(StrEnum):
    STRATIFIED = "stratified"
    RANDOM = "random"


class ProviderKind(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class SystemConfig(_Base):
    """Where the scores of the system under test come from."""

    adapter: AdapterKind = AdapterKind.JSONL
    path: Path = Path("./evals/results/")


class AnchorConfig(_Base):
    """The control group. The system under test never touches these items."""

    mode: AnchorMode = AnchorMode.FROZEN_SELF
    n: Annotated[int, Field(ge=1)] = 260
    cadence: Annotated[int, Field(ge=1)] = 1
    selection: SelectionKind = SelectionKind.STRATIFIED
    noise_replicates: Annotated[int, Field(ge=2)] = 5
    #: Pinned so stratified selection is reproducible (Phase 3.4).
    seed: Annotated[int, Field(ge=0)] = 0
    #: Optional path to gold labels; only meaningful in `human` mode.
    labels: Path | None = None

    @model_validator(mode="after")
    def _labels_only_in_human_mode(self) -> Self:
        if self.labels is not None and self.mode is not AnchorMode.HUMAN:
            raise ValueError(
                f"`labels` is only used in `human` mode, but mode is `{self.mode.value}`"
            )
        if self.labels is None and self.mode is AnchorMode.HUMAN:
            raise ValueError("`human` mode requires `labels:` pointing at gold labels")
        return self


class JudgeParams(_Base):
    """Sampling parameters. Any change here is a judge change (Hard Rule 8)."""

    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.0
    max_tokens: Annotated[int, Field(ge=1)] = 512
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    seed: int | None = None
    response_format: str | None = None


class JudgeConfig(_Base):
    provider: ProviderKind = ProviderKind.ANTHROPIC
    model: Annotated[str, Field(min_length=1)] = "claude-sonnet-4-5-20250929"
    rubric: Path = Path("./evals/rubric.md")
    params: JudgeParams = JudgeParams()
    #: Phase 7.9 — defeat provider-side response caching on the anchor stream.
    cache_busting_nonce: bool = False


class VerdictName(StrEnum):
    """String form of the verdicts, used in gate configuration."""

    STABLE = "stable"
    JUDGE = "judge"
    SYSTEM = "system"
    BOTH = "both"
    INDETERMINATE = "indeterminate"


class GateConfig(_Base):
    """`fail_on: [system, both]` is the product in one line of YAML: a judge change
    must not fail your build, it must tell you to re-baseline."""

    fail_on: tuple[VerdictName, ...] = (VerdictName.SYSTEM, VerdictName.BOTH)
    warn_on: tuple[VerdictName, ...] = (VerdictName.INDETERMINATE,)

    @model_validator(mode="after")
    def _no_overlap(self) -> Self:
        both = set(self.fail_on) & set(self.warn_on)
        if both:
            names = ", ".join(sorted(v.value for v in both))
            raise ValueError(f"verdict(s) appear in both fail_on and warn_on: {names}")
        if VerdictName.STABLE in self.fail_on:
            raise ValueError("`stable` in fail_on would fail every healthy build")
        return self


class StatsConfig(_Base):
    """Knobs on the sequential machinery. Defaults are the ones we validated."""

    #: Bounded memory for the e-detector (Hard Rule 4). Pruning may only delay detection.
    max_candidates: Annotated[int, Field(ge=1)] = 256
    #: Betting fraction truncation, keeps wealth strictly positive.
    bet_truncation: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    #: Runs used to estimate the baseline mean before monitoring starts.
    baseline_runs: Annotated[int, Field(ge=1)] = 8
    #: Runs within which a judge shift must be detectable, for provisioning.
    horizon: Annotated[int, Field(ge=1)] = 50


class BenchlockConfig(_Base):
    """The whole of `benchlock.yaml`."""

    version: Literal[1] = SCHEMA_VERSION  # type: ignore[assignment]
    alpha: Annotated[float, Field(gt=0.0, le=0.5)] = 0.05
    score_scale: tuple[float, float] = (0.0, 1.0)
    min_runs: Annotated[int, Field(ge=2)] = 8
    min_obs: Annotated[int, Field(ge=1)] = 30
    system: SystemConfig = SystemConfig()
    anchor: AnchorConfig = AnchorConfig()
    judge: JudgeConfig = JudgeConfig()
    gate: GateConfig = GateConfig()
    stats: StatsConfig = StatsConfig()

    @model_validator(mode="after")
    def _scale_is_ordered(self) -> Self:
        lo, hi = self.score_scale
        if not hi > lo:
            raise ValueError(f"score_scale must be [low, high] with high > low, got [{lo}, {hi}]")
        return self

    # ---- construction -----------------------------------------------------------------

    @classmethod
    def parse(cls, text: str, source: str = DEFAULT_CONFIG_NAME) -> BenchlockConfig:
        """Parse YAML text into a config, raising ConfigError with line numbers."""
        data, lines = load_yaml_with_lines(text, source)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            issues = [
                ConfigIssue(
                    loc=tuple(err["loc"]),
                    line=_line_for(tuple(err["loc"]), lines),
                    message=_humanise(dict(err)),
                    hint=_hint_for(tuple(err["loc"])),
                )
                for err in exc.errors()
            ]
            raise ConfigError(source, issues) from exc

    @classmethod
    def load(cls, path: Path) -> BenchlockConfig:
        if not path.exists():
            raise ConfigError(
                str(path),
                [
                    ConfigIssue(
                        loc=(),
                        line=0,
                        message="config file not found",
                        hint="run `benchlock init` in your eval project to create one",
                    )
                ],
            )
        return cls.parse(path.read_text(), source=str(path))

    def to_yaml(self) -> str:
        """Dump to YAML. `load -> dump -> load` is stable (Phase 0.2 verify)."""
        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def discover_config_path(start: Path | None = None) -> Path | None:
    """`./benchlock.yaml`, then `$XDG_CONFIG_HOME/benchlock/config.yaml` (§5)."""
    for candidate in _candidate_config_paths(start):
        if candidate.exists():
            return candidate
    return None


def _candidate_config_paths(start: Path | None = None) -> Iterator[Path]:
    import os

    yield (start or Path.cwd()) / DEFAULT_CONFIG_NAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    yield base / "benchlock" / "config.yaml"
