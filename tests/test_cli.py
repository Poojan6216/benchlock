"""Phase 0.2 verify: the CLI surface exists and fails loudly rather than silently."""

from __future__ import annotations

import json

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


#: Commands whose backends land in later phases. Shrinks as the build progresses; a
#: command that disappears from here has been implemented, and its own suite covers it.
NOT_YET = ["replay", "report", "plan", "baseline"]


@pytest.mark.parametrize("command", NOT_YET)
def test_unimplemented_commands_fail_loudly(command: str) -> None:
    # Never a silent no-op: exit 3, and the message names the phase it arrives in.
    result = runner.invoke(app, [command])
    assert result.exit_code == EXIT_ERROR
    assert "not implemented yet" in result.output


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
