"""Phase 8.5 verify: the eight mistakes a real person makes on their first afternoon.

Every one must produce a message that names the file, the field and the command that
fixes it. None may produce a stack trace. The bar is not "it errors" — it is that someone
who has never read the source can act on what they see.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchlock.cli import EXIT_OK, app

runner = CliRunner()

GOOD_CONFIG = """\
version: 1
alpha: 0.05
score_scale: [1, 5]
min_runs: 8
min_obs: 30
judge:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  rubric: ./rubric.md
"""


def project(tmp_path: Path, config: str = GOOD_CONFIG, *, rubric: bool = True) -> Path:
    (tmp_path / "benchlock.yaml").write_text(config)
    if rubric:
        (tmp_path / "rubric.md").write_text("Score the answer 1-5.\n")
    return tmp_path


def invoke(project_dir: Path, *args: str):
    return runner.invoke(app, [*args, "--config", str(project_dir / "benchlock.yaml")])


def assert_actionable(result, *, must_contain: str, command: str | None = None) -> None:
    assert result.exit_code != EXIT_OK, "this should have failed"
    assert "Traceback" not in result.output, f"a stack trace escaped:\n{result.output}"
    assert must_contain in result.output, (
        f"expected {must_contain!r} in the message, got:\n{result.output}"
    )
    if command is not None:
        assert command in result.output, (
            f"the message should name the fixing command {command!r}:\n{result.output}"
        )


# --- the eight most likely mistakes ------------------------------------------------------


def test_1_running_verdict_before_recording_anything(tmp_path: Path) -> None:
    result = invoke(project(tmp_path), "verdict", "--ledger", str(tmp_path / "nope.jsonl"))
    assert_actionable(result, must_contain="no ledger", command="benchlock observe")


def test_2_no_config_at_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    result = runner.invoke(app, ["verdict"])
    assert_actionable(result, must_contain="no benchlock.yaml", command="benchlock init")


def test_3_a_typo_in_the_config(tmp_path: Path) -> None:
    # A misspelled key, not a duplicate one: YAML legally overwrites duplicates, so
    # `min_runs: 12` twice is not a typo the config layer can see.
    root = project(tmp_path, GOOD_CONFIG + "min_runss: 12\n")
    result = invoke(root, "verdict")
    assert_actionable(result, must_contain="unknown field")
    assert "benchlock.yaml:" in result.output, "the error must name the line"
    assert "min_runss" in result.output


def test_4_forgetting_to_declare_the_score_scale(tmp_path: Path) -> None:
    """The default is [0,1]; a 1-5 rubric then reads as wildly out of range."""
    root = project(tmp_path, GOOD_CONFIG.replace("score_scale: [1, 5]\n", ""))
    results = tmp_path / "run.jsonl"
    results.write_text('{"item_id": "a", "score": 4}\n{"item_id": "b", "score": 5}\n')
    result = invoke(root, "observe", str(results), "--ledger", str(tmp_path / "l.jsonl"))
    assert_actionable(result, must_contain="outside the declared score_scale")
    assert "score_scale" in result.output
    assert "will not clamp" in result.output


def test_5_pointing_at_a_file_that_does_not_exist(tmp_path: Path) -> None:
    result = invoke(
        project(tmp_path),
        "observe",
        str(tmp_path / "missing.jsonl"),
        "--ledger",
        str(tmp_path / "l.jsonl"),
    )
    assert_actionable(result, must_contain="file not found")


def test_6_the_rubric_path_is_wrong(tmp_path: Path) -> None:
    """The rubric is hashed into the judge pin, so it has to be readable."""
    root = project(tmp_path, rubric=False)
    results = tmp_path / "run.jsonl"
    results.write_text("\n".join(f'{{"item_id": "q{i}", "score": 4}}' for i in range(40)))
    result = invoke(root, "observe", str(results), "--ledger", str(tmp_path / "l.jsonl"))
    assert_actionable(result, must_contain="judge rubric not found")
    assert "judge.rubric" in result.output


def test_7_planning_before_measuring_a_noise_floor(tmp_path: Path) -> None:
    """`plan` needs the judge's measured self-disagreement; it will not assume one."""
    result = invoke(project(tmp_path), "plan", "--ledger", str(tmp_path / "l.jsonl"))
    assert_actionable(result, must_contain="no anchor baseline", command="benchlock baseline")
    assert "measured rather than assumed" in result.output


def test_8_too_few_items_in_a_run(tmp_path: Path) -> None:
    root = project(tmp_path)
    results = tmp_path / "small.jsonl"
    results.write_text("\n".join(f'{{"item_id": "q{i}", "score": 4}}' for i in range(5)))
    result = invoke(root, "observe", str(results), "--ledger", str(tmp_path / "l.jsonl"))
    assert_actionable(result, must_contain="below min_obs")
    assert "refused rather than silently monitored" in result.output


# --- and the paths that should just work --------------------------------------------------


def test_init_on_an_empty_directory_still_produces_a_usable_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == EXIT_OK
    assert (tmp_path / "benchlock.yaml").exists()
    assert "next:" in result.output, "init must say what to do next"


def test_every_error_path_names_a_command(tmp_path: Path) -> None:
    """Across the first-use failures, every message points at a next step."""
    cases = [
        invoke(project(tmp_path), "verdict", "--ledger", str(tmp_path / "a.jsonl")),
        invoke(project(tmp_path), "plan", "--ledger", str(tmp_path / "b.jsonl")),
        invoke(project(tmp_path), "replay", "--ledger", str(tmp_path / "c.jsonl")),
        invoke(
            project(tmp_path), "rebaseline", "--reason", "x", "--ledger", str(tmp_path / "d.jsonl")
        ),
    ]
    for result in cases:
        assert "benchlock " in result.output, f"no command suggested:\n{result.output}"
        assert "Traceback" not in result.output


def test_help_is_available_for_every_command() -> None:
    import typer.main

    for name in typer.main.get_command(app).commands:  # type: ignore[attr-defined]
        result = runner.invoke(app, [name, "--help"])
        assert result.exit_code == EXIT_OK
        assert len(result.output.splitlines()) > 3, f"{name} has no useful help"
