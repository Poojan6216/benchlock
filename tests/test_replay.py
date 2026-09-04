"""Phase 4 verify: every historical verdict re-derives, and a change to the rules is caught."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bench.sim.generate import StreamSpec, generate
from typer.testing import CliRunner

from benchlock.attribute.engine import decide
from benchlock.cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, app
from benchlock.config import AttributionConfig
from benchlock.ledger.log import Ledger, RecordType
from benchlock.report.markdown import render_markdown

runner = CliRunner()
CONFIG = AttributionConfig(baseline_runs=8, target_shift=0.05)
RUBRIC = "Score the answer 1-5 for helpfulness and factual accuracy.\n"

PROJECT = """\
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


def build_ledger(tmp_path: Path, spec: StreamSpec) -> Path:
    """A project with a ledger carrying a verdict recorded after every run."""
    (tmp_path / "rubric.md").write_text(RUBRIC)
    (tmp_path / "benchlock.yaml").write_text(PROJECT)

    system, anchor = generate(spec)
    book = Ledger(tmp_path / ".benchlock" / "ledger.jsonl")
    book.append_baseline(
        judge_pin=anchor[0].judge_pin, anchor_pin=anchor[0].anchor_pin, note="test"
    )
    for i, (s, a) in enumerate(zip(system, anchor, strict=True)):
        book.append_run(s)
        book.append_run(a)
        attribution = decide(system[: i + 1], anchor[: i + 1], CONFIG)
        book.append_verdict(
            attribution.to_json(), at_system_run=s.run_index, at_anchor_run=a.run_index
        )
    return tmp_path


def invoke(project: Path, *args: str) -> object:
    return runner.invoke(
        app,
        [
            *args,
            "--config",
            str(project / "benchlock.yaml"),
            "--ledger",
            str(project / ".benchlock" / "ledger.jsonl"),
        ],
    )


# --- 4.1 replay ---------------------------------------------------------------------------


@pytest.mark.mandatory
def test_replay_reproduces_every_historical_verdict(tmp_path: Path) -> None:
    project = build_ledger(
        tmp_path, StreamSpec(name="replay", seed=99, n_runs=60, change_at=25, judge_shift=-0.09)
    )
    result = invoke(project, "replay")
    assert result.exit_code == EXIT_OK, result.output
    assert "60 historical verdict(s) re-derived exactly" in result.output


@pytest.mark.mandatory
@pytest.mark.slow
def test_replay_reproduces_all_three_hundred_verdicts(tmp_path: Path) -> None:
    """The spec's 300-run ledger, as its own case."""
    project = build_ledger(
        tmp_path, StreamSpec(name="long", seed=99, n_runs=300, change_at=120, judge_shift=-0.09)
    )
    book = Ledger(project / ".benchlock" / "ledger.jsonl")
    assert len([r for r in book.verify() if r.type is RecordType.VERDICT]) == 300

    result = invoke(project, "replay", "--json")
    payload = json.loads(result.output)
    assert payload["checked"] == 300
    assert payload["ok"] is True
    assert payload["divergences"] == []
    assert result.exit_code == EXIT_OK


@pytest.mark.mandatory
def test_changing_a_lattice_constant_makes_replay_fail_and_names_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spec's negative control: alter the rules, and history stops reproducing.

    Without this, `replay` could be a green light that only ever confirms today's code
    agrees with itself.
    """
    project = build_ledger(
        tmp_path, StreamSpec(name="tamper", seed=7, n_runs=40, change_at=15, system_shift=-0.06)
    )
    assert invoke(project, "replay").exit_code == EXIT_OK

    # Move the decision boundary: everything now looks like it crossed.
    import benchlock.attribute.lattice as lattice

    original = lattice.apply_lattice

    def shifted(evidence, alpha, **kwargs):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        return original(replace(evidence, threshold=evidence.threshold / 1000.0), alpha, **kwargs)

    monkeypatch.setattr("benchlock.attribute.engine.apply_lattice", shifted)

    result = invoke(project, "replay")
    assert result.exit_code == EXIT_FAIL
    assert "no longer reproduce" in result.output
    assert "recorded" in result.output and "replayed" in result.output
    # The first divergent run is named, not just the count.
    assert "system run" in result.output


def test_replay_detects_a_tampered_ledger_before_it_starts(tmp_path: Path) -> None:
    project = build_ledger(tmp_path, StreamSpec(name="t", seed=3, n_runs=20))
    path = project / ".benchlock" / "ledger.jsonl"
    lines = path.read_text().splitlines()
    target = next(
        i for i, line in enumerate(lines) if json.loads(line)["type"] == RecordType.RUN.value
    )
    data = json.loads(lines[target])
    data["payload"]["observations"][0]["score"] = 0.123
    lines[target] = json.dumps(data, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    result = invoke(project, "replay")
    assert result.exit_code == EXIT_ERROR
    assert f"at record {target}" in result.output


def test_replay_on_a_missing_ledger_is_actionable(tmp_path: Path) -> None:
    (tmp_path / "rubric.md").write_text(RUBRIC)
    (tmp_path / "benchlock.yaml").write_text(PROJECT)
    result = invoke(tmp_path, "replay")
    assert result.exit_code == EXIT_ERROR
    assert "benchlock observe" in result.output


# --- 4.2 versioned decision semantics -------------------------------------------------------


def test_a_semantics_change_reports_both_verdicts_rather_than_rewriting_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old verdicts were not wrong. They were issued under different rules."""
    project = build_ledger(
        tmp_path, StreamSpec(name="v", seed=11, n_runs=40, change_at=15, system_shift=-0.06)
    )
    # Rewrite the recorded semantics version to v1 while the code claims v2.
    path = project / ".benchlock" / "ledger.jsonl"
    monkeypatch.setattr("benchlock.DECISION_SEMANTICS_VERSION", 2)

    import benchlock.attribute.lattice as lattice

    original = lattice.apply_lattice

    def v2(evidence, alpha, **kwargs):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        return original(replace(evidence, threshold=evidence.threshold / 1000.0), alpha, **kwargs)

    monkeypatch.setattr("benchlock.attribute.engine.apply_lattice", v2)

    result = invoke(project, "replay", "--json")
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["semantics_version"] == 2
    divergence = payload["divergences"][0]
    assert divergence["recorded_semantics"] == 1
    assert divergence["replayed_semantics"] == 2
    assert divergence["semantics_changed"] is True
    # Both verdicts are reported side by side.
    assert divergence["recorded"]["verdict"] != divergence["replayed"]["verdict"]
    assert result.exit_code == EXIT_FAIL
    assert path.exists(), "replay must never rewrite the ledger"


def test_every_ledger_record_carries_its_semantics_version(tmp_path: Path) -> None:
    project = build_ledger(tmp_path, StreamSpec(name="s", seed=5, n_runs=12))
    for record in Ledger(project / ".benchlock" / "ledger.jsonl").verify():
        assert record.semantics_version >= 1


# --- 4.3 report ---------------------------------------------------------------------------------


def test_report_renders_markdown(tmp_path: Path) -> None:
    project = build_ledger(
        tmp_path, StreamSpec(name="r", seed=13, n_runs=40, change_at=15, system_shift=-0.06)
    )
    result = invoke(project, "report")
    assert result.exit_code == EXIT_OK
    body = result.output
    assert body.startswith("### ")
    assert "| process | e-value | threshold | status |" in body
    assert "|---|---:|---:|---|" in body, "GitHub needs the alignment row"
    assert "`E_anchor`" in body and "`E_corrected`" in body
    assert "```sh" in body


def test_report_writes_to_a_file(tmp_path: Path) -> None:
    project = build_ledger(tmp_path, StreamSpec(name="rf", seed=15, n_runs=20))
    out = tmp_path / "pr" / "comment.md"
    result = invoke(project, "report", "--out", str(out))
    assert result.exit_code == EXIT_OK
    assert out.read_text().startswith("### ")


def test_markdown_matches_the_verdict_it_was_rendered_from(tmp_path: Path) -> None:
    """A PR comment and a terminal must never disagree about what happened."""
    system, anchor = generate(
        StreamSpec(name="md", seed=17, n_runs=40, change_at=15, system_shift=-0.06)
    )
    attribution = decide(system, anchor, CONFIG)
    body = render_markdown(attribution)
    assert f"`{attribution.verdict.value}`" in body
    assert f"rule `{attribution.rule_id}`" in body
    assert f"{attribution.evidence.e_system:,.1f}" in body


def test_markdown_code_fences_are_balanced(tmp_path: Path) -> None:
    system, anchor = generate(StreamSpec(name="f", seed=19, n_runs=40, system_shift=-0.06))
    body = render_markdown(decide(system, anchor, CONFIG))
    assert body.count("```") % 2 == 0, "an unbalanced fence swallows the rest of the comment"


def test_markdown_tables_have_consistent_column_counts() -> None:
    system, anchor = generate(StreamSpec(name="t2", seed=21, n_runs=40, system_shift=-0.06))
    body = render_markdown(decide(system, anchor, CONFIG))
    table: list[str] = []
    for line in body.splitlines():
        if line.startswith("|"):
            table.append(line)
        elif table:
            widths = {row.count("|") for row in table}
            assert len(widths) == 1, f"ragged markdown table: {table}"
            table = []
