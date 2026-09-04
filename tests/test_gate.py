"""Phase 2.6 verify: each verdict maps to the right exit code, under default and custom config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bench.sim.generate import generate, read_manifest, spec_from_json
from typer.testing import CliRunner

from benchlock.cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, EXIT_WARN, app
from benchlock.ledger.log import Ledger

runner = CliRunner()
MANIFEST = Path(__file__).resolve().parent / "fixtures" / "streams" / "manifest.json"

CONFIG = """\
version: 1
alpha: 0.05
score_scale: [1, 5]
min_runs: 8
min_obs: 30
judge:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  rubric: ./rubric.md
gate:
  fail_on: {fail_on}
  warn_on: {warn_on}
"""


def build_project(
    tmp_path: Path,
    fixture: str,
    *,
    fail_on: str = "[system, both]",
    warn_on: str = "[indeterminate]",
) -> Path:
    """A real project directory with a real ledger built from a golden stream."""
    entry = next(e for e in read_manifest(MANIFEST) if e["name"] == fixture)
    system, anchor = generate(spec_from_json(entry))

    (tmp_path / "rubric.md").write_text("Score the answer 1-5.\n")
    (tmp_path / "benchlock.yaml").write_text(CONFIG.format(fail_on=fail_on, warn_on=warn_on))

    book = Ledger(tmp_path / ".benchlock" / "ledger.jsonl")
    book.append_baseline(
        judge_pin=anchor[0].judge_pin, anchor_pin=anchor[0].anchor_pin, note="test"
    )
    for s, a in zip(system, anchor, strict=True):
        book.append_run(s)
        book.append_run(a)
    return tmp_path


def run_gate(project: Path, *args: str) -> object:
    return runner.invoke(
        app,
        [
            "gate",
            "--config",
            str(project / "benchlock.yaml"),
            "--ledger",
            str(project / ".benchlock" / "ledger.jsonl"),
            *args,
        ],
    )


# --- default gate configuration ---------------------------------------------------------


DEFAULTS = [
    ("stable-control", EXIT_OK, "stable"),
    # The product in one line of YAML: a judge change must NOT fail your build.
    ("demo1-phantom-judge", EXIT_OK, "judge"),
    ("demo2-real-regression", EXIT_FAIL, "system"),
    ("both-moved", EXIT_FAIL, "both"),
    ("demo3-under-provisioned", EXIT_WARN, "indeterminate"),
]


@pytest.mark.parametrize(("fixture", "code", "verdict"), DEFAULTS, ids=[d[0] for d in DEFAULTS])
def test_default_gate_exit_codes(tmp_path: Path, fixture: str, code: int, verdict: str) -> None:
    project = build_project(tmp_path, fixture)
    result = run_gate(project)
    assert f"verdict={verdict}" in result.output, result.output
    assert result.exit_code == code, f"{fixture} ({verdict}) exited {result.exit_code}"


def test_a_judge_change_does_not_fail_the_build() -> None:
    """Stated as its own test because it is the entire product thesis."""
    assert {d[2]: d[1] for d in DEFAULTS}["judge"] == EXIT_OK


# --- custom gate configuration ------------------------------------------------------------


def test_indeterminate_can_be_made_to_fail(tmp_path: Path) -> None:
    project = build_project(
        tmp_path, "demo3-under-provisioned", fail_on="[system, both, indeterminate]", warn_on="[]"
    )
    assert run_gate(project).exit_code == EXIT_FAIL


def test_a_team_can_choose_to_be_told_about_judge_drift(tmp_path: Path) -> None:
    project = build_project(
        tmp_path, "demo1-phantom-judge", fail_on="[system, both]", warn_on="[judge, indeterminate]"
    )
    result = run_gate(project)
    assert result.exit_code == EXIT_WARN
    assert "WARN" in result.output


def test_nothing_configured_to_fail_passes_everything(tmp_path: Path) -> None:
    project = build_project(tmp_path, "demo2-real-regression", fail_on="[]", warn_on="[]")
    assert run_gate(project).exit_code == EXIT_OK


# --- verdict command ----------------------------------------------------------------------


def test_verdict_prints_the_block(tmp_path: Path) -> None:
    project = build_project(tmp_path, "demo2-real-regression")
    result = runner.invoke(
        app,
        [
            "verdict",
            "--config",
            str(project / "benchlock.yaml"),
            "--ledger",
            str(project / ".benchlock" / "ledger.jsonl"),
        ],
    )
    assert result.exit_code == EXIT_OK
    assert "verdict=system" in result.output
    assert "E_corrected" in result.output


def test_verdict_json_is_machine_readable(tmp_path: Path) -> None:
    project = build_project(tmp_path, "demo2-real-regression")
    result = runner.invoke(
        app,
        [
            "verdict",
            "--json",
            "--config",
            str(project / "benchlock.yaml"),
            "--ledger",
            str(project / ".benchlock" / "ledger.jsonl"),
        ],
    )
    payload = json.loads(result.output)
    assert payload["verdict"] == "system"
    assert payload["rule_id"] == "system_only"
    assert payload["evidence"]["e_corrected"] > payload["evidence"]["threshold"]
    assert payload["reasons"]


# --- failure modes ---------------------------------------------------------------------------


def test_gate_on_a_missing_ledger_explains_what_to_do(tmp_path: Path) -> None:
    (tmp_path / "rubric.md").write_text("x")
    (tmp_path / "benchlock.yaml").write_text(
        CONFIG.format(fail_on="[system]", warn_on="[indeterminate]")
    )
    result = run_gate(tmp_path)
    assert result.exit_code == EXIT_ERROR
    assert "benchlock observe" in result.output
    assert "Traceback" not in result.output


def test_gate_refuses_across_an_undeclared_pin_change(tmp_path: Path) -> None:
    project = build_project(tmp_path, "declared-judge-swap")
    result = run_gate(project)
    assert result.exit_code == EXIT_ERROR
    assert "without a logged rebaseline" in result.output
    assert "benchlock rebaseline" in result.output
