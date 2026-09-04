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

import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from benchlock import __version__
from benchlock.adapters.jsonl import IngestError
from benchlock.adapters.jsonl import load as load_jsonl
from benchlock.config import DEFAULT_CONFIG_NAME, BenchlockConfig, ConfigError, discover_config_path
from benchlock.jsonlog import Level, log
from benchlock.ledger.log import Ledger, LedgerError, new_run_id
from benchlock.model.pins import JudgePin
from benchlock.model.streams import RunRecord, StreamKind, suite_hash_of

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
    _todo("init", "Phase 0.6")


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

        run = RunRecord(
            run_id=new_run_id(),
            run_index=len(existing) + appended,
            kind=stream,
            observations=observations,
            suite_hash=suite_hash_of(o.item_id for o in observations),
            judge_pin=_judge_pin(cfg),
            anchor_pin=None,
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
) -> None:
    """Attribute the current score movement: judge, system, both, neither, or unknown."""
    _todo("verdict", "Phase 2.3")


@app.command()
def gate(
    config: ConfigOpt = None,
    ledger: LedgerOpt = DEFAULT_LEDGER,
) -> None:
    """Print the verdict and exit 0/1/2 per gate.fail_on. The last line of a CI job."""
    _todo("gate", "Phase 2.6")


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
