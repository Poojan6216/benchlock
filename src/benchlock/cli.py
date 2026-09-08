"""The `benchlock` command line.

Nine subcommands: init, plan, baseline, observe, verdict, gate, replay, rebaseline, report.

Exit codes are part of the contract, because `benchlock gate` is designed to be the last
line of a CI job:

    0   verdict is allowed (default: stable, judge)
    1   verdict fails the build (default: system, both)
    2   verdict warns (default: indeterminate)
    3   the tool could not run at all — bad config, broken ledger, pin violation
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from benchlock import __version__
from benchlock.adapters import Detection, detect_framework
from benchlock.adapters.jsonl import IngestError
from benchlock.adapters.jsonl import load as load_jsonl
from benchlock.anchor.coverage import measure_coverage
from benchlock.anchor.modes import (
    AnchorItem,
    AnchorModeError,
    check_mode_supported,
    freeze,
    load_anchors,
    save_anchors,
)
from benchlock.anchor.select import Candidate, select_anchors
from benchlock.attribute.engine import decide
from benchlock.config import (
    DEFAULT_CONFIG_NAME,
    AdapterKind,
    AttributionConfig,
    BenchlockConfig,
    ConfigError,
    ProviderKind,
    discover_config_path,
)
from benchlock.jsonlog import Level, log
from benchlock.judge.base import JudgeAdapter, SimulatedJudge, estimate_cost_for
from benchlock.ledger.log import Ledger, LedgerError, new_run_id
from benchlock.ledger.replay import replay as replay_ledger
from benchlock.model.pins import (
    JudgePin,
    PinViolationError,
    check_anchor_pin,
    check_judge_pin,
)
from benchlock.model.streams import Observation, RunRecord, StreamKind, suite_hash_of
from benchlock.model.verdict import Attribution, AttributionRefusedError
from benchlock.report.human import render_verdict_block
from benchlock.report.markdown import render_markdown
from benchlock.stats.power import ProvisioningImpossibleError, make_plan

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_WARN = 2
EXIT_ERROR = 3

app = typer.Typer(
    name="benchlock",
    help=(
        "Did your system regress, or did the judge change underneath you?\n\n"
        "Benchlock holds a frozen anchor set your system never touches, watches both "
        "streams with anytime-valid sequential tests, and attributes score movement to "
        "the judge, the system, both, or neither — or refuses to attribute at all."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help=f"Path to {DEFAULT_CONFIG_NAME}.", show_default=False),
]
LedgerOpt = Annotated[
    Path,
    typer.Option("--ledger", help="Path to the append-only ledger."),
]
DEFAULT_LEDGER = Path(".benchlock/ledger.jsonl")
TargetShiftOpt = Annotated[
    float,
    typer.Option(
        "--target-shift",
        help="The judge shift the anchor set was provisioned to catch. Sets the monitoring band.",
    ),
]


def _die(message: str, hint: str | None = None, code: int = EXIT_ERROR) -> NoReturn:
    """Fail loudly, on stderr, naming the fix (Hard Rule 10)."""
    log.error("benchlock.failed", message=message, hint=hint)
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    if hint:
        typer.secho(f"  fix: {hint}", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code)


def _load_config(path: Path | None) -> BenchlockConfig:
    resolved = path or discover_config_path()
    if resolved is None:
        _die(
            f"no {DEFAULT_CONFIG_NAME} found in this directory",
            "run `benchlock init` to create one from your existing eval setup",
        )
    try:
        cfg = BenchlockConfig.load(resolved)
    except ConfigError as exc:
        typer.secho(exc.render(), fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    cfg = _resolve_paths(cfg, resolved.parent)
    log.debug("config.loaded", path=str(resolved), alpha=cfg.alpha)
    return cfg


def _resolve_paths(cfg: BenchlockConfig, root: Path) -> BenchlockConfig:
    """Make relative paths relative to the *config file*, not the working directory.

    A committed `benchlock.yaml` says `rubric: ./evals/rubric.md`, and that has to mean
    the same thing whether CI runs from the repo root or a subdirectory. Resolving
    against the config's own directory is the only reading that survives being checked in.
    """

    def under(path: Path) -> Path:
        return path if path.is_absolute() else (root / path)

    return cfg.model_copy(
        update={
            "judge": cfg.judge.model_copy(update={"rubric": under(cfg.judge.rubric)}),
            "system": cfg.system.model_copy(update={"path": under(cfg.system.path)}),
            "anchor": cfg.anchor.model_copy(
                update={"labels": under(cfg.anchor.labels) if cfg.anchor.labels else None}
            ),
        }
    )


def _judge_pin(cfg: BenchlockConfig, *, simulate: bool = False) -> JudgePin:
    """Hash the judge configuration. The rubric must be readable: it is part of the pin.

    The pin comes from the adapter that would actually do the scoring, never rebuilt
    beside it from the config. Those two are not the same thing: a config that sets no
    `judge.params` still produces requests carrying the adapter's own defaults, so a pin
    built from the config alone describes an instrument that was never used. Building it
    twice is how a ledger ends up with a `params_hash` that no run can reproduce.

    `--simulate` is part of the judge's identity, not a testing convenience bolted on
    beside it: a simulated judge and a hosted one are different measuring instruments, so
    switching between them must trip the pin check exactly like any other judge change.
    """
    return _judge_adapter(cfg, simulate=simulate).pin()


def _relativise(path: Path, root: Path) -> Path:
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        return path
    return Path("./") / rel if rel.parts else Path("./")


def _render_config(found: Detection) -> str:
    """Render a commented benchlock.yaml. Comments matter: this is the file a user reads
    to understand what the tool is asking of them."""
    scale = found.score_scale or (0.0, 1.0)
    scale_comment = (
        f"  # {found.score_scale_note}"
        if found.score_scale_note
        else "  # REQUIRED: set this to your rubric's range"
    )
    evidence = "\n".join(f"# - {line}" for line in found.evidence)
    return f"""\
# Written by `benchlock init`. What it found in this project:
{evidence}
version: 1

# The false-alarm budget for the whole monitoring process, not per run.
alpha: 0.05

# The range your rubric emits. Benchlock normalises to [0,1] internally and keeps the
# raw value; an out-of-range score is an error, not a clamp.
score_scale: [{scale[0]:g}, {scale[1]:g}]{scale_comment}

min_runs: 8     # no verdict is attempted before this many runs
min_obs: 30     # minimum judged items in a run

system:
  adapter: {found.adapter.value}
  path: {found.path}

anchor:
  # The control group: items your system never touches, re-scored by the judge each run.
  # If these move, only the judge can have moved them.
  mode: frozen-self           # needs no human labels
  n: 260                      # size it properly with `benchlock plan --target-shift 0.05`
  cadence: 1
  selection: stratified
  noise_replicates: 5
  seed: 0

judge:
  provider: anthropic
  model: {found.judge_model or "claude-sonnet-5"}
  rubric: ./evals/rubric.md   # the exact text is hashed into the judge pin
  # No `params` on purpose: the adapter's defaults are the ones that work with the model
  # above. Current models reject `temperature`/`top_p` with a 400. Anything you set here
  # is hashed into the judge pin, so changing it later needs `benchlock rebaseline`.
  cache_busting_nonce: false   # true if your provider caches identical judge prompts

gate:
  # A judge change must NOT fail your build. It must tell you to re-baseline.
  fail_on: [system, both]
  warn_on: [indeterminate]
"""


def _attribute(cfg: BenchlockConfig, ledger_path: Path, target_shift: float) -> Attribution:
    """Load the ledger and decide. Shared by `verdict`, `gate` and `report`."""
    book = Ledger(ledger_path)
    if not book.exists():
        _die(
            f"no ledger at {ledger_path}",
            "record some runs first: `benchlock observe <your-eval-output.jsonl>`",
        )
    try:
        system = book.runs(StreamKind.SYSTEM)
        anchor = book.runs(StreamKind.ANCHOR)
    except LedgerError as exc:
        _die(exc.message, exc.hint or "restore the ledger from version control")

    if not system:
        _die(
            "the ledger holds no system runs yet",
            "run `benchlock observe <your-eval-output.jsonl>` after each eval run",
        )

    try:
        return decide(system, anchor, AttributionConfig.from_config(cfg, target_shift=target_shift))
    except AttributionRefusedError as exc:
        _die(exc.message, exc.hint)


def _record_verdict(book: Ledger, attribution: Attribution) -> None:
    """Append the verdict to the ledger so `benchlock replay` has something to check.

    An audit trail of inputs alone would let replay confirm that today's code agrees with
    itself while saying nothing about what the tool actually told you last quarter.
    """
    system = book.runs(StreamKind.SYSTEM)
    anchor = book.runs(StreamKind.ANCHOR)
    book.append_verdict(
        attribution.to_json(),
        at_system_run=system[-1].run_index if system else 0,
        at_anchor_run=anchor[-1].run_index if anchor else 0,
    )


def _exit_code_for(verdict: str, cfg: BenchlockConfig) -> int:
    """gate.fail_on is the product in one line of YAML: `judge` must not fail your build."""
    if verdict in {v.value for v in cfg.gate.fail_on}:
        return EXIT_FAIL
    if verdict in {v.value for v in cfg.gate.warn_on}:
        return EXIT_WARN
    return EXIT_OK


def _judge_adapter(
    cfg: BenchlockConfig, *, simulate: bool = False, seed_offset: int = 0
) -> JudgeAdapter:
    """The judge to score with. Real providers land in Phase 6; `--simulate` needs no key.

    `seed_offset` varies the simulated judge's noise between runs. Without it every
    invocation rebuilds the same generator and replays the identical noise, which would
    make the anchor stream perfectly stable — a simulation flattering itself rather than
    exercising the thing being tested.
    """
    if simulate:
        rubric = cfg.judge.rubric
        return SimulatedJudge(
            seed=cfg.anchor.seed + seed_offset,
            scale=cfg.score_scale,
            model=f"simulated::{cfg.judge.model}",
            rubric_text=rubric.read_text(encoding="utf-8") if rubric.exists() else "rubric",
        )
    rubric = cfg.judge.rubric
    if not rubric.exists():
        _die(
            f"judge rubric not found at {rubric}",
            "point `judge.rubric` in benchlock.yaml at the rubric/system prompt your judge "
            "uses; its exact text is hashed into the judge pin",
        )
    rubric_text = rubric.read_text(encoding="utf-8")
    # Only what the config actually sets. An empty dict means "use the adapter's own
    # defaults", which are the ones chosen to work with the models that adapter names.
    params = cfg.judge.params.model_dump(mode="json", exclude_none=True)

    if cfg.judge.provider is ProviderKind.ANTHROPIC:
        try:
            from benchlock.judge.anthropic import AnthropicJudge
        except ImportError:  # pragma: no cover - exercised by the extras, not the suite
            _die(
                "the `anthropic` package is not installed",
                "install the optional extra: `uv pip install 'benchlock[anthropic]'`, or "
                "pass --simulate to use the built-in judge, which needs no key",
            )
        anthropic_judge = AnthropicJudge(
            model=cfg.judge.model, rubric_text=rubric_text, scale=cfg.score_scale
        )
        if params:
            anthropic_judge.params = params
        return anthropic_judge

    if cfg.judge.provider is ProviderKind.OPENAI:
        try:
            from benchlock.judge.openai import OpenAIJudge
        except ImportError:  # pragma: no cover - exercised by the extras, not the suite
            _die(
                "the `openai` package is not installed",
                "install the optional extra: `uv pip install 'benchlock[openai]'`, or "
                "pass --simulate to use the built-in judge, which needs no key",
            )
        openai_judge = OpenAIJudge(
            model=cfg.judge.model, rubric_text=rubric_text, scale=cfg.score_scale
        )
        if params:
            openai_judge.params = params
        return openai_judge

    _die(  # pragma: no cover - ProviderKind has no third member
        f"no judge adapter is wired for provider `{cfg.judge.provider.value}`",
        "pass --simulate to use the deterministic built-in judge, which needs no API key",
    )


def _ingest(cfg: BenchlockConfig, path: Path) -> tuple[Observation, ...]:
    """Read one run's scores with the adapter the config actually names.

    `benchlock init` detects promptfoo / Inspect AI / DeepEval and writes the answer into
    `system.adapter`. Reading every file as JSONL regardless would make that detection
    worse than useless: a valid promptfoo export would be reported back to the user as
    malformed JSON, one error per line, naming their file rather than our dispatch.
    """
    scale = cfg.score_scale
    if cfg.system.adapter is AdapterKind.PROMPTFOO:
        from benchlock.adapters.promptfoo import load as load_promptfoo

        return load_promptfoo(path, scale)
    if cfg.system.adapter is AdapterKind.INSPECT_AI:
        from benchlock.adapters.inspect_ai import load as load_inspect

        return load_inspect(path, scale)
    if cfg.system.adapter is AdapterKind.DEEPEVAL:
        from benchlock.adapters.deepeval import load as load_deepeval

        return load_deepeval(path, scale)
    return load_jsonl(path, scale)


AnchorStoreOpt = Annotated[
    Path, typer.Option("--anchor-store", help="Where the frozen anchor pairs are kept.")
]
DEFAULT_ANCHOR_STORE = Path(".benchlock/anchors.jsonl")


def _anchor_candidates(
    cfg: BenchlockConfig, path: Path | None, store: Path = DEFAULT_ANCHOR_STORE
) -> list[AnchorItem]:
    """Load the pairs to freeze: an explicit file, or the existing frozen set."""
    if path is not None:
        if not path.exists():
            _die(
                f"anchor source not found at {path}",
                'pass a JSONL of {"item_id", "prompt_input", "output", "tags"} objects',
            )
        items: list[AnchorItem] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                _die(f"{path}:{line_no}: not valid JSON: {exc.msg}", "one JSON object per line")
            if "item_id" not in data:
                _die(f"{path}:{line_no}: no `item_id`", "every anchor item needs a stable id")
            items.append(
                AnchorItem(
                    item_id=str(data["item_id"]),
                    prompt_input=str(data.get("prompt_input", "")),
                    output=str(data.get("output", "")),
                    tags=tuple(data.get("tags", ())),
                    gold=data.get("gold"),
                )
            )
        if not items:
            _die(
                f"{path} contains no anchor items", "the file should hold one JSON object per line"
            )
        return items
    try:
        return load_anchors(store)
    except AnchorModeError as exc:
        _die(exc.message, exc.hint)


def _rescore_and_record(
    book: Ledger,
    cfg: BenchlockConfig,
    *,
    simulate: bool,
    run_index: int,
    store: Path = DEFAULT_ANCHOR_STORE,
) -> None:
    """Re-score the frozen anchor set for this run and append it as an anchor run.

    This is what makes the control group a *control*: the same frozen outputs, judged
    again, every run. Without it the anchor stream stops at the baseline and nothing can
    tell you the judge moved.

    ``run_index`` is the index of the SYSTEM run this scoring accompanies, not a counter
    over anchor records. The difference-in-differences stream joins the two legs on
    ``run_index`` (`engine._corrected_stream`), so numbering anchor runs independently
    silently lags the control leg by the K noise-floor replicates — every paired
    comparison would then contrast a system run against an anchor scoring taken K runs
    earlier, which is the one thing the design exists to avoid.
    """
    from benchlock.anchor.modes import FrozenAnchor, rescore

    _, anchor_pin = book.current_pins()
    if anchor_pin is None:
        _die(
            "no anchor baseline has been frozen, so there is nothing to re-score",
            "run `benchlock baseline` first",
        )
    try:
        items = load_anchors(store)
    except AnchorModeError as exc:
        _die(exc.message, exc.hint)

    existing_anchor_runs = len(book.runs(StreamKind.ANCHOR))
    judge = _judge_adapter(cfg, simulate=simulate, seed_offset=existing_anchor_runs + 1)
    nonce = new_run_id() if cfg.judge.cache_busting_nonce else ""
    try:
        scores = rescore(items, judge, nonce=nonce)
    except AnchorModeError as exc:
        _die(exc.message, exc.hint)

    # The frozen set must still be the frozen set. `observe --kind anchor` checks this;
    # this path is the one CI actually uses (`--rescore-anchors`, and the shipped GitHub
    # Action defaults it on), and an anchor run scored on a different item set reads as
    # exactly zero movement rather than as an error — a judge shift then looks like a
    # system regression. Same check, same rule (Hard Rule 8).
    observed = replace(
        anchor_pin,
        item_set_hash=suite_hash_of(sorted(scores)),
        n=len(scores),
    )
    try:
        check_anchor_pin(observed, anchor_pin)
    except PinViolationError as exc:
        _die(exc.message, exc.hint)

    frozen = FrozenAnchor(
        pin=anchor_pin,
        baseline_scores={},
        noise=None,  # type: ignore[arg-type]
        judge_pin=_judge_pin(cfg, simulate=simulate),
        mode=anchor_pin.mode,
    )
    _append_anchor_run(book, cfg, scores, frozen, run_index=run_index)
    mean = sum(scores.values()) / len(scores)
    typer.echo(f"  re-scored {len(scores)} anchor items: mean {mean:.4f}")


def _append_anchor_run(
    book: Ledger,
    cfg: BenchlockConfig,
    scores: Mapping[str, float],
    frozen: object,
    *,
    run_index: int,
) -> None:
    """Record one anchor scoring as a run. Scores arrive already normalised to [0,1]."""
    from benchlock.anchor.modes import FrozenAnchor

    assert isinstance(frozen, FrozenAnchor)
    lo, hi = cfg.score_scale
    observations = tuple(
        Observation(item_id=item, score=score, raw_score=lo + score * (hi - lo), scale=(lo, hi))
        for item, score in sorted(scores.items())
    )
    book.append_run(
        RunRecord(
            run_id=new_run_id(),
            run_index=run_index,
            kind=StreamKind.ANCHOR,
            observations=observations,
            suite_hash=suite_hash_of(o.item_id for o in observations),
            judge_pin=frozen.judge_pin,
            anchor_pin=frozen.pin,
            epoch=book.epoch(),
        )
    )


def _observations_per_run(book: Ledger, cfg: BenchlockConfig) -> int:
    """How many items a system run scores. Measured from the ledger where possible."""
    if book.exists():
        runs = book.runs(StreamKind.SYSTEM)
        if runs:
            return runs[-1].n
    return cfg.min_obs


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"benchlock {__version__}")
        raise typer.Exit(EXIT_OK)


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logs on stderr.")] = False,
) -> None:
    if verbose:
        log.level = Level.DEBUG


# ---------------------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------------------


@app.command()
def init(
    directory: Annotated[Path, typer.Argument(help="Project to scan.")] = Path(),
    force: Annotated[bool, typer.Option("--force", help="Overwrite (a backup is kept).")] = False,
) -> None:
    """Detect your eval framework and write a pre-filled benchlock.yaml."""
    root = directory.resolve()
    if not root.is_dir():
        _die(f"{directory} is not a directory", "point `benchlock init` at your eval project")

    found = detect_framework(root)
    # Paths are written relative to the project so the config stays valid once committed.
    found = replace(found, path=_relativise(found.path, root))
    target = root / DEFAULT_CONFIG_NAME

    if target.exists():
        # Never overwrite without a byte-for-byte backup, even with --force.
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup = target.with_suffix(f".yaml.{stamp}.bak")
        backup.write_bytes(target.read_bytes())
        typer.echo(f"backed up existing config to {backup.name}")
        if not force:
            _die(
                f"{DEFAULT_CONFIG_NAME} already exists",
                f"a backup was written to {backup.name}; re-run with --force to replace it",
            )

    text = _render_config(found)
    # A generated config that does not parse is worse than none at all.
    try:
        BenchlockConfig.parse(text, source=str(target))
    except ConfigError as exc:  # pragma: no cover - guards a template regression
        _die("generated config failed its own validation", exc.render())
    target.write_text(text, encoding="utf-8")

    typer.echo(f"wrote {target.relative_to(Path.cwd()) if root == Path.cwd() else target}")
    for line in found.evidence:
        typer.echo(f"  - {line}")
    if found.score_scale_note:
        typer.secho(f"  ! score_scale {found.score_scale_note}", fg=typer.colors.YELLOW)
    if not found.confident:
        typer.secho(
            "  ! no eval framework was detected — set `system.path` to your score output",
            fg=typer.colors.YELLOW,
        )
    typer.echo("")
    typer.echo("next:")
    typer.echo("  1. confirm `score_scale` matches your rubric's range")
    typer.echo("  2. point `judge.rubric` at your rubric/system prompt")
    # `plan` reads the noise floor that `baseline` measures, so recommending it first
    # sends a first-time user straight into an error on the very command we told them
    # to run. Baseline comes first, and says why.
    typer.echo("  3. benchlock baseline --anchors your-suite.jsonl")
    typer.echo("       freezes the anchors and measures the noise floor")
    typer.echo("  4. benchlock plan --target-shift 0.05")
    typer.echo("       sizes the anchor set against that measured floor")


@app.command()
def plan(
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
    target_shift: Annotated[
        float, typer.Option("--target-shift", help="Smallest judge shift you must attribute.")
    ] = 0.05,
    horizon: Annotated[
        int | None, typer.Option("--horizon", help="Runs within which detection must occur.")
    ] = None,
) -> None:
    """Size the anchor set: how many items, how often, and what it will cost."""
    cfg = _load_config(config)
    book = Ledger(ledger)
    _, anchor_pin = book.current_pins() if book.exists() else (None, None)
    if anchor_pin is None:
        _die(
            "no anchor baseline has been frozen, so there is no measured noise floor "
            "to plan against",
            "run `benchlock baseline` first — provisioning depends on how much your judge "
            "disagrees with itself, which has to be measured rather than assumed",
        )

    runs = horizon if horizon is not None else cfg.stats.horizon
    obs_per_run = _observations_per_run(book, cfg)
    try:
        plan_result = make_plan(
            target_shift,
            anchor_pin.noise_floor,
            cfg.alpha,
            runs,
            obs_per_run,
            cadence=cfg.anchor.cadence,
        )
    except ProvisioningImpossibleError as exc:
        _die(exc.message, exc.hint)

    judge = _judge_adapter(cfg, simulate=True)
    per_run = estimate_cost_for(judge.describe(), plan_result.anchor_n)
    baseline_cost = estimate_cost_for(
        judge.describe(), plan_result.anchor_n * cfg.anchor.noise_replicates
    )

    typer.echo(
        f"target shift      {target_shift:.3f}   (the smallest judge move you must attribute)"
    )
    typer.echo(f"horizon           {runs} runs")
    typer.echo(f"alpha             {cfg.alpha}")
    typer.echo("")
    typer.secho(f"anchor n >= {plan_result.anchor_n}", bold=True)
    typer.echo(f"cadence           every {plan_result.cadence} run(s)")
    typer.echo(
        f"achieved          min detectable judge shift "
        f"{plan_result.achieved_min_detectable_shift:.4f}"
    )
    typer.echo(
        f"dead zone         {plan_result.dead_zone:.4f}  "
        f"(snapshot error at K={plan_result.replicates}; no shift smaller is ever detectable)"
    )
    typer.echo("")
    typer.echo(
        f"cost per run      {per_run.input_tokens + per_run.output_tokens:,} tokens, "
        f"${per_run.dollars:.3f}"
    )
    typer.echo(
        f"cost to baseline  {baseline_cost.input_tokens + baseline_cost.output_tokens:,} "
        f"tokens, ${baseline_cost.dollars:.3f}  "
        f"({cfg.anchor.noise_replicates} replicates)"
    )
    if plan_result.shared_noise_dominates:
        typer.secho(
            f"\n! quadrupling the anchor set would improve the detectable shift by only "
            f"{plan_result.marginal_gain_at_4x:.0%}. Your judge's run-to-run movement is "
            f"shared across items ({plan_result.shared_sd:.4f}), and that does not shrink "
            "with more anchors. More replicates or a steadier judge will help; more items "
            "will not",
            fg=typer.colors.YELLOW,
        )
    if plan_result.anchor_n > cfg.anchor.n:
        typer.secho(
            f"\n! benchlock.yaml has anchor.n = {cfg.anchor.n}, below the {plan_result.anchor_n} "
            "this target needs. Verdicts will come back `indeterminate` rather than `system`",
            fg=typer.colors.YELLOW,
        )


@app.command()
def baseline(
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
    anchors: Annotated[
        Path | None,
        typer.Option("--anchors", help="JSONL of anchor pairs to freeze."),
    ] = None,
    anchor_store: AnchorStoreOpt = DEFAULT_ANCHOR_STORE,
    simulate: Annotated[
        bool,
        typer.Option("--simulate", help="Use the deterministic built-in judge (no API key)."),
    ] = False,
) -> None:
    """Freeze the anchor set and measure the judge's noise floor."""
    cfg = _load_config(config)
    judge = _judge_adapter(cfg, simulate=simulate)
    try:
        check_mode_supported(cfg.anchor.mode, judge)
    except AnchorModeError as exc:
        _die(exc.message, exc.hint)

    items = _anchor_candidates(cfg, anchors, anchor_store)
    suite = [Candidate(item_id=i.item_id, score=0.5, tags=i.tags) for i in items]
    if len(items) > cfg.anchor.n:
        selection = select_anchors(
            suite, cfg.anchor.n, seed=cfg.anchor.seed, kind=cfg.anchor.selection
        )
        keep = set(selection.chosen)
        items = [i for i in items if i.item_id in keep]
        typer.echo(
            f"selected {len(items)} of {len(suite)} suite items "
            f"({cfg.anchor.selection.value}, seed {cfg.anchor.seed})"
        )

    typer.echo(
        f"scoring {len(items)} anchor items x {cfg.anchor.noise_replicates} replicates "
        f"with {judge.describe().provider}/{judge.describe().model}..."
    )
    try:
        # The nonce has to reach the replicates too. The noise floor is measured by asking
        # the judge the SAME question K times; behind a provider that caches responses,
        # K byte-identical prompts return one cached answer K times and the floor comes
        # back as exactly zero — after which any later movement reads as drift. Measuring
        # the floor with a cacheable prompt is the one place a cache does most damage.
        frozen = freeze(
            items,
            judge,
            replicates=cfg.anchor.noise_replicates,
            mode=cfg.anchor.mode,
            nonce_prefix=new_run_id() if cfg.judge.cache_busting_nonce else "",
        )
    except AnchorModeError as exc:
        _die(exc.message, exc.hint)

    save_anchors(items, anchor_store)
    book = Ledger(ledger)
    book.append_baseline(
        judge_pin=frozen.judge_pin,
        anchor_pin=frozen.pin,
        epoch=book.epoch(),
        note=f"{cfg.anchor.mode.value} anchor set of {frozen.pin.n} items",
    )
    # The K replicates go in as the first K anchor runs, so the snapshot the verdict is
    # measured against can be rebuilt from the ledger and re-derived by `replay`.
    for index, scoring in enumerate(frozen.replicate_scorings):
        _append_anchor_run(book, cfg, scoring, frozen, run_index=index)

    coverage = measure_coverage(suite, [i.item_id for i in items])
    typer.echo("")
    typer.echo(f"anchor set frozen: {frozen.pin.n} items, mode {frozen.mode.value}")
    typer.echo(f"  noise floor: per-item SD {frozen.noise.floor.per_item_sd:.4f}, ")
    typer.echo(f"               run-mean SD {frozen.noise.floor.run_mean_sd:.5f}")
    typer.echo(
        f"  judge self-agreement: {frozen.noise.exact_agreement_rate:.1%} of identical calls "
        f"returned an identical score"
    )
    typer.echo(f"  stored at {anchor_store} (eval content — keep it out of version control)")
    for warning in (*frozen.warnings(), *coverage.warnings()):
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)
    typer.echo("")
    typer.echo("next:")
    typer.echo("  benchlock plan --target-shift 0.05    # check this anchor set is big enough")


# ---------------------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------------------


@app.command()
def observe(
    results: Annotated[list[Path], typer.Argument(help="Eval output file(s) to ingest.")],
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
    kind: Annotated[str, typer.Option("--kind", help="system | anchor")] = "system",
    rescore_anchors: Annotated[
        bool,
        typer.Option(
            "--rescore-anchors",
            help="Also re-score the frozen anchor set with the judge and record it.",
        ),
    ] = False,
    anchor_store: AnchorStoreOpt = DEFAULT_ANCHOR_STORE,
    simulate: Annotated[
        bool, typer.Option("--simulate", help="Use the deterministic built-in judge.")
    ] = False,
) -> None:
    """Ingest one run's scores and append them to the ledger."""
    cfg = _load_config(config)
    try:
        stream = StreamKind(kind)
    except ValueError:
        _die(f"unknown --kind {kind!r}", "use `--kind system` or `--kind anchor`")

    book = Ledger(ledger)
    try:
        existing = book.runs(stream)
    except LedgerError as exc:
        _die(exc.message, exc.hint or "restore the ledger from version control")

    if stream is StreamKind.ANCHOR and not any(r.type.value == "baseline" for r in book.read_raw()):
        _die(
            "no anchor baseline has been frozen yet",
            "run `benchlock baseline` to freeze the anchor set and measure the noise floor",
        )

    # Hard Rule 8: compare the judge we are about to record against the one in force.
    pinned_judge, pinned_anchor = book.current_pins()
    current_judge = _judge_pin(cfg, simulate=simulate)
    if pinned_judge is not None:
        try:
            check_judge_pin(current_judge, pinned_judge)
        except PinViolationError as exc:
            _die(exc.message, exc.hint)

    for appended, path in enumerate(results):
        try:
            observations = _ingest(cfg, path)
        except IngestError as exc:
            typer.secho(exc.render(), fg=typer.colors.RED, err=True)
            raise typer.Exit(EXIT_ERROR) from exc

        if len(observations) < cfg.min_obs:
            _die(
                f"{path} has {len(observations)} observations, below min_obs={cfg.min_obs}",
                "either lower `min_obs` in benchlock.yaml or score more items; a run too "
                "small to be informative is refused rather than silently monitored",
            )

        run_anchor_pin = None
        if stream is StreamKind.ANCHOR and pinned_anchor is not None:
            # The anchor set is frozen: a run that scores a different set of items is not
            # a measurement of the same control group (Hard Rule 8).
            observed = replace(
                pinned_anchor,
                item_set_hash=suite_hash_of(o.item_id for o in observations),
                n=len(observations),
            )
            try:
                check_anchor_pin(observed, pinned_anchor)
            except PinViolationError as exc:
                _die(exc.message, exc.hint)
            run_anchor_pin = pinned_anchor

        run = RunRecord(
            run_id=new_run_id(),
            run_index=len(existing) + appended,
            kind=stream,
            observations=observations,
            suite_hash=suite_hash_of(o.item_id for o in observations),
            judge_pin=current_judge,
            anchor_pin=run_anchor_pin,
            epoch=book.epoch(),
        )
        try:
            record = book.append_run(run)
        except LedgerError as exc:
            _die(exc.message, exc.hint)
        log.info(
            "observe.appended",
            path=str(path),
            kind=stream.value,
            run_index=run.run_index,
            n=run.n,
            mean=round(run.mean, 6),
            seq=record.seq,
        )
        typer.echo(
            f"recorded run {run.run_index} ({stream.value}): "
            f"{run.n} items, mean {run.mean:.4f} -> {ledger}"
        )

        if rescore_anchors and stream is StreamKind.SYSTEM:
            # `anchor.cadence` is what the user set to control judge spend, and what
            # `benchlock plan` provisioned against. Ignoring it here meant a team that set
            # cadence 5 to cut anchor scoring to a fifth paid the full anchor set on every
            # CI run, with nothing in the output saying the setting had been dropped.
            # Skipped runs simply yield fewer paired runs, which the corrected stream
            # already handles by intersecting on run_index.
            if run.run_index % cfg.anchor.cadence == 0:
                _rescore_and_record(
                    book, cfg, simulate=simulate, run_index=run.run_index, store=anchor_store
                )
            else:
                nxt = (run.run_index // cfg.anchor.cadence + 1) * cfg.anchor.cadence
                typer.echo(
                    f"  anchors not re-scored: `anchor.cadence` is {cfg.anchor.cadence}, "
                    f"so the next anchor run is system run {nxt}"
                )


@app.command()
def verdict(
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
    json_out: Annotated[bool, typer.Option("--json", help="Emit the Attribution as JSON.")] = False,
    target_shift: TargetShiftOpt = 0.05,
    record: Annotated[
        bool, typer.Option("--record/--no-record", help="Append the verdict to the ledger.")
    ] = True,
) -> None:
    """Attribute the current score movement: judge, system, both, neither, or unknown."""
    cfg = _load_config(config)
    attribution = _attribute(cfg, ledger, target_shift)
    if json_out:
        typer.echo(json.dumps(attribution.to_json(), indent=2))
    else:
        typer.echo(render_verdict_block(attribution), nl=False)
    if record:
        _record_verdict(Ledger(ledger), attribution)
    log.info(
        "verdict.decided",
        verdict=attribution.verdict.value,
        rule=attribution.rule_id,
        e_system=round(attribution.evidence.e_system, 3),
        e_anchor=round(attribution.evidence.e_anchor, 3),
    )


@app.command()
def gate(
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
    target_shift: TargetShiftOpt = 0.05,
) -> None:
    """Print the verdict and exit 0/1/2 per gate.fail_on. The last line of a CI job."""
    cfg = _load_config(config)
    attribution = _attribute(cfg, ledger, target_shift)
    typer.echo(render_verdict_block(attribution), nl=False)

    _record_verdict(Ledger(ledger), attribution)
    code = _exit_code_for(attribution.verdict.value, cfg)
    log.info(
        "gate.decided", verdict=attribution.verdict.value, exit_code=code, rule=attribution.rule_id
    )
    if code == EXIT_FAIL:
        typer.secho(f"gate: FAIL — verdict `{attribution.verdict.value}`", fg=typer.colors.RED)
    elif code == EXIT_WARN:
        typer.secho(f"gate: WARN — verdict `{attribution.verdict.value}`", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"gate: PASS — verdict `{attribution.verdict.value}`", fg=typer.colors.GREEN)
    raise typer.Exit(code)


# ---------------------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------------------


@app.command()
def replay(
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
    target_shift: TargetShiftOpt = 0.05,
    json_out: Annotated[bool, typer.Option("--json", help="Emit the report as JSON.")] = False,
) -> None:
    """Re-derive every historical verdict from the ledger. A mismatch is a failure."""
    cfg = _load_config(config)
    book = Ledger(ledger)
    if not book.exists():
        _die(f"no ledger at {ledger}", "record some runs first with `benchlock observe`")
    try:
        report_result = replay_ledger(
            book, AttributionConfig.from_config(cfg, target_shift=target_shift)
        )
    except LedgerError as exc:
        _die(exc.message, exc.hint or "restore the ledger from version control")

    if json_out:
        typer.echo(json.dumps(report_result.to_json(), indent=2))
    elif report_result.ok:
        typer.secho(
            f"replay OK: {report_result.checked} historical verdict(s) re-derived exactly "
            f"under decision semantics v{report_result.semantics_version}",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            f"replay FAILED: {len(report_result.divergences)} of {report_result.checked} "
            "historical verdict(s) no longer reproduce",
            fg=typer.colors.RED,
        )
        for divergence in report_result.divergences:
            typer.echo(f"  {divergence.describe()}")
        for refusal in report_result.refusals:
            typer.echo(f"  {refusal}")
        first = report_result.first_divergence
        if first is not None and first.semantics_changed:
            typer.secho(
                "\nThe decision semantics version changed, so both verdicts are reported "
                "rather than history being rewritten to agree with the new code. If the "
                "change was intended, this listing is the changelog for it.",
                fg=typer.colors.YELLOW,
            )
    log.info(
        "replay.finished",
        checked=report_result.checked,
        ok=report_result.ok,
        divergences=len(report_result.divergences),
    )
    if not report_result.ok:
        raise typer.Exit(EXIT_FAIL)


@app.command()
def rebaseline(
    reason: Annotated[str, typer.Option("--reason", help="Why. Required, and logged.")],
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
) -> None:
    """Start a new baseline epoch. Explicit, logged, versioned (Hard Rule 8)."""
    cfg = _load_config(config)
    book = Ledger(ledger)
    if not book.exists():
        _die(
            f"no ledger at {ledger}",
            "there is nothing to rebaseline; run `benchlock observe` first",
        )
    try:
        pinned_judge, pinned_anchor = book.current_pins()
    except LedgerError as exc:
        _die(exc.message, exc.hint)

    current_judge = _judge_pin(cfg)
    delta = current_judge.differs_from(pinned_judge) if pinned_judge else ()
    new_epoch = book.epoch() + 1
    record = book.append_rebaseline(
        reason=reason,
        epoch=new_epoch,
        judge_pin=current_judge,
        anchor_pin=pinned_anchor,
        pin_delta=delta,
    )
    log.info("rebaseline.recorded", epoch=new_epoch, reason=reason, seq=record.seq, delta=delta)
    typer.echo(f"epoch {new_epoch} started: {reason}")
    if delta:
        typer.echo(f"  judge pin fields that moved: {', '.join(delta)}")
    typer.echo(
        "  history before this point is retained and replayable, and no verdict will "
        "compare across the boundary"
    )


@app.command()
def report(
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
    out: Annotated[Path | None, typer.Option("--out", help="Write markdown here.")] = None,
    target_shift: TargetShiftOpt = 0.05,
) -> None:
    """Markdown for a PR comment: the verdict block, traces, and provisioning status."""
    cfg = _load_config(config)
    attribution = _attribute(cfg, ledger, target_shift)
    markdown = render_markdown(attribution)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        typer.echo(f"wrote {out}")
    else:
        typer.echo(markdown, nl=False)


def main() -> None:
    try:
        app()
    except ConfigError as exc:  # pragma: no cover - defence in depth
        typer.secho(exc.render(), fg=typer.colors.RED, err=True)
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":  # pragma: no cover
    main()
