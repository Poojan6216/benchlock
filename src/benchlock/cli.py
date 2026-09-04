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
from benchlock.config import DEFAULT_CONFIG_NAME, BenchlockConfig, ConfigError, discover_config_path
from benchlock.jsonlog import Level, log

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
    _todo("observe", "Phase 0.4")


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
