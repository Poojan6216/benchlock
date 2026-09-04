"""Phase 0.2 verify: the CLI surface exists and fails loudly rather than silently."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchlock.cli import EXIT_ERROR, EXIT_OK, app
from benchlock.config import BenchlockConfig

runner = CliRunner()

NINE_COMMANDS = [
    "init",
    "plan",
    "baseline",
    "observe",
    "verdict",
    "gate",
    "replay",
    "rebaseline",
    "report",
]


def test_all_nine_subcommands_are_present() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == EXIT_OK
    # Typer leaves `name=None` when it derives the name from the function, so ask click.
    import typer.main

    registered = set(typer.main.get_command(app).commands)  # type: ignore[attr-defined]
    assert registered == set(NINE_COMMANDS), f"unexpected command set: {sorted(registered)}"


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == EXIT_OK
    assert "benchlock" in result.stdout


@pytest.mark.parametrize("command", NINE_COMMANDS)
def test_every_command_has_help(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == EXIT_OK
    assert command in result.stdout


def test_every_command_is_implemented() -> None:
    """All nine have real backends now; none is a stub that silently does nothing.

    Kept as a regression guard: a command reintroduced as a stub would slip past the
    help-text tests, which only check that it exists.
    """
    from benchlock import cli

    source = Path(inspect.getfile(cli)).read_text()
    assert "_todo(" not in source, "a command was left as a stub"


@pytest.mark.parametrize("command", ["verdict", "gate", "replay", "report"])
def test_commands_fail_loudly_without_a_ledger(command: str, tmp_path: Path) -> None:
    """Never a stack trace: a missing ledger explains what to do about it."""
    (tmp_path / "rubric.md").write_text("Score 1-5.\n")
    (tmp_path / "benchlock.yaml").write_text(
        "version: 1\nscore_scale: [1, 5]\n"
        "judge:\n  provider: anthropic\n  model: m\n  rubric: ./rubric.md\n"
    )
    result = runner.invoke(
        app,
        [
            command,
            "--config",
            str(tmp_path / "benchlock.yaml"),
            "--ledger",
            str(tmp_path / ".benchlock" / "ledger.jsonl"),
        ],
    )
    assert result.exit_code == EXIT_ERROR
    assert "Traceback" not in result.output
    assert "benchlock observe" in result.output


def test_no_args_prints_help() -> None:
    result = runner.invoke(app, [])
    assert "Commands" in result.output


def test_the_example_config_is_valid() -> None:
    # benchlock.yaml.example is the thing users copy; it must parse.
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "benchlock.yaml.example"
    cfg = BenchlockConfig.parse(example.read_text(), source=str(example))
    assert cfg.score_scale == (1.0, 5.0)
    assert cfg.gate.fail_on == ("system", "both")
    # The one line that is the product: a judge change must not fail CI.
    assert "judge" not in cfg.gate.fail_on


def test_logs_are_json_on_stderr() -> None:
    import io

    from benchlock.jsonlog import Logger

    buf = io.StringIO()
    Logger(buf, timestamps=False).info("thing.happened", runs=3, verdict="stable")
    record = json.loads(buf.getvalue())
    assert record == {"level": "info", "event": "thing.happened", "runs": 3, "verdict": "stable"}
