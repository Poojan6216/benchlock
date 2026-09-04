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
from dataclasses import replace
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from benchlock import __version__
from benchlock.adapters import Detection, detect_framework
from benchlock.adapters.jsonl import IngestError
from benchlock.adapters.jsonl import load as load_jsonl
from benchlock.attribute.engine import decide
from benchlock.config import (
    DEFAULT_CONFIG_NAME,
    AttributionConfig,
    BenchlockConfig,
    ConfigError,
    discover_config_path,
)
from benchlock.jsonlog import Level, log
from benchlock.ledger.log import Ledger, LedgerError, new_run_id
from benchlock.model.pins import JudgePin, PinViolationError, check_anchor_pin, check_judge_pin
from benchlock.model.streams import RunRecord, StreamKind, suite_hash_of
from benchlock.model.verdict import Attribution, AttributionRefusedError
from benchlock.report.human import render_verdict_block

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
    log.debug("config.loaded", path=str(resolved), alpha=cfg.alpha)
    return cfg


def _judge_pin(cfg: BenchlockConfig) -> JudgePin:
    """Hash the judge configuration. The rubric must be readable: it is part of the pin."""
    rubric = cfg.judge.rubric
    if not rubric.exists():
        _die(
            f"judge rubric not found at {rubric}",
            "point `judge.rubric` in benchlock.yaml at the rubric/system prompt your judge "
            "uses; its exact text is hashed into the judge pin",
        )
    return JudgePin.build(
        provider=cfg.judge.provider.value,
        model=cfg.judge.model,
        rubric_text=rubric.read_text(encoding="utf-8"),
        params=cfg.judge.params.model_dump(mode="json", exclude_none=True),
        scale=cfg.score_scale,
    )


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
  model: {found.judge_model or "claude-sonnet-4-5-20250929"}
  rubric: ./evals/rubric.md   # the exact text is hashed into the judge pin
  params:
    temperature: 0.0
    max_tokens: 512

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


def _exit_code_for(verdict: str, cfg: BenchlockConfig) -> int:
    """gate.fail_on is the product in one line of YAML: `judge` must not fail your build."""
    if verdict in {v.value for v in cfg.gate.fail_on}:
        return EXIT_FAIL
    if verdict in {v.value for v in cfg.gate.warn_on}:
        return EXIT_WARN
    return EXIT_OK


def _todo(task: str, phase: str) -> NoReturn:
    """A subcommand whose backend lands in a later phase. Never silently no-ops."""
    _die(
        f"`benchlock {task}` is not implemented yet (arrives in {phase})",
        "see BUILD_SPEC.md for the build order",
    )


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
    typer.echo("  3. benchlock plan --target-shift 0.05    # size the anchor set")


@app.command()
def plan(
    config: ConfigOpt = None,
    target_shift: Annotated[
        float, typer.Option("--target-shift", help="Smallest judge shift you must attribute.")
    ] = 0.05,
    horizon: Annotated[
        int | None, typer.Option("--horizon", help="Runs within which detection must occur.")
    ] = None,
) -> None:
    """Size the anchor set: how many items, how often, and what it will cost."""
    _todo("plan", "Phase 3.6")


@app.command()
def baseline(
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
) -> None:
    """Freeze the anchor set and measure the judge's noise floor."""
    _todo("baseline", "Phase 3.1")


# ---------------------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------------------


@app.command()
def observe(
    results: Annotated[list[Path], typer.Argument(help="Eval output file(s) to ingest.")],
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
    kind: Annotated[str, typer.Option("--kind", help="system | anchor")] = "system",
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
    current_judge = _judge_pin(cfg)
    if pinned_judge is not None:
        try:
            check_judge_pin(current_judge, pinned_judge)
        except PinViolationError as exc:
            _die(exc.message, exc.hint)

    for appended, path in enumerate(results):
        try:
            observations = load_jsonl(path, cfg.score_scale)
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


@app.command()
def verdict(
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
    json_out: Annotated[bool, typer.Option("--json", help="Emit the Attribution as JSON.")] = False,
    target_shift: TargetShiftOpt = 0.05,
) -> None:
    """Attribute the current score movement: judge, system, both, neither, or unknown."""
    cfg = _load_config(config)
    attribution = _attribute(cfg, ledger, target_shift)
    if json_out:
        typer.echo(json.dumps(attribution.to_json(), indent=2))
    else:
        typer.echo(render_verdict_block(attribution), nl=False)
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
) -> None:
    """Re-derive every historical verdict from the ledger. A mismatch is a failure."""
    _todo("replay", "Phase 4.1")


@app.command()
def rebaseline(
    reason: Annotated[str, typer.Option("--reason", help="Why. Required, and logged.")],
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
) -> None:
    """Start a new baseline epoch. Explicit, logged, versioned (Hard Rule 8)."""
    _todo("rebaseline", "Phase 3.7")


@app.command()
def report(
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
    out: Annotated[Path | None, typer.Option("--out", help="Write markdown here.")] = None,
) -> None:
    """Markdown for a PR comment: the verdict block, traces, and provisioning status."""
    _todo("report", "Phase 4.3")


def main() -> None:
    try:
        app()
    except ConfigError as exc:  # pragma: no cover - defence in depth
        typer.secho(exc.render(), fg=typer.colors.RED, err=True)
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":  # pragma: no cover
    main()
