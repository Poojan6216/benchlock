"""Phase 3.7 verify: rebaselining is explicit, logged, and isolates epochs."""

from __future__ import annotations

from pathlib import Path

import pytest
from bench.sim.generate import StreamSpec, generate
from typer.testing import CliRunner

from benchlock.attribute.engine import decide
from benchlock.cli import EXIT_ERROR, EXIT_OK, app
from benchlock.config import AttributionConfig
from benchlock.ledger.log import Ledger, RecordType
from benchlock.model.verdict import Verdict

runner = CliRunner()
CONFIG = AttributionConfig(baseline_runs=8, target_shift=0.05)

PROJECT_CONFIG = """\
version: 1
alpha: 0.05
score_scale: [1, 5]
min_runs: 8
min_obs: 30
judge:
  provider: anthropic
  model: {model}
  rubric: ./rubric.md
"""


#: Must match the rubric the stream generator hashes into its pins, so that changing the
#: model string is the *only* difference between the two configs below.
GENERATOR_RUBRIC = "Score the answer 1-5 for helpfulness and factual accuracy.\n"


def build(tmp_path: Path, model: str = "claude-sonnet-4-5-20250929") -> Path:
    (tmp_path / "rubric.md").write_text(GENERATOR_RUBRIC)
    (tmp_path / "benchlock.yaml").write_text(PROJECT_CONFIG.format(model=model))
    return tmp_path


def seed_ledger(book: Ledger, spec: StreamSpec, *, epoch: int = 0) -> None:
    system, anchor = generate(spec)
    for s, a in zip(system, anchor, strict=True):
        for run in (s, a):
            book.append_run(type(run)(**{**_fields(run), "epoch": epoch}))


def _fields(run: object) -> dict:
    return {f: getattr(run, f) for f in run.__dataclass_fields__}  # type: ignore[attr-defined]


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


def test_rebaseline_requires_a_reason() -> None:
    result = runner.invoke(app, ["rebaseline"])
    assert result.exit_code != EXIT_OK
    assert "reason" in result.output.lower()


def test_rebaseline_on_an_empty_ledger_explains_itself(tmp_path: Path) -> None:
    project = build(tmp_path)
    result = invoke(project, "rebaseline", "--reason", "why-not")
    assert result.exit_code == EXIT_ERROR
    assert "benchlock observe" in result.output


def test_rebaseline_records_an_epoch_and_names_the_pin_delta(tmp_path: Path) -> None:
    project = build(tmp_path)
    book = Ledger(project / ".benchlock" / "ledger.jsonl")
    seed_ledger(book, StreamSpec(name="e0", seed=1, n_runs=14, judge_shift=-0.1))
    assert book.epoch() == 0

    # The provider rotated the snapshot and the team updated the config to match.
    build(project, model="claude-sonnet-4-5-20260114")
    result = invoke(project, "rebaseline", "--reason", "judge-version-change")
    assert result.exit_code == EXIT_OK
    assert "epoch 1 started: judge-version-change" in result.output
    assert "judge pin fields that moved: model" in result.output

    assert book.epoch() == 1
    records = book.verify()
    rebaselines = [r for r in records if r.type is RecordType.REBASELINE]
    assert len(rebaselines) == 1
    assert rebaselines[0].payload["reason"] == "judge-version-change"
    assert rebaselines[0].payload["pin_delta"] == ["model"]


def test_verdicts_never_compare_across_an_epoch_boundary(tmp_path: Path) -> None:
    """History before a rebaseline is retained and replayable, never silently mixed in."""
    project = build(tmp_path)
    book = Ledger(project / ".benchlock" / "ledger.jsonl")

    # Epoch 0: a judge that drifted badly. Epoch 1: a clean stream under the new baseline.
    seed_ledger(book, StreamSpec(name="old", seed=1, n_runs=40, judge_shift=-0.2), epoch=0)
    seed_ledger(book, StreamSpec(name="new", seed=2, n_runs=40), epoch=1)

    from benchlock.model.streams import StreamKind

    system = book.runs(StreamKind.SYSTEM)
    anchor = book.runs(StreamKind.ANCHOR)
    assert len(system) == 80, "the old epoch is still in the ledger"

    result = decide(system, anchor, CONFIG)
    assert result.evidence.epoch == 1
    assert result.evidence.n_runs == 40, "only the current epoch is decided over"
    assert result.verdict is Verdict.STABLE, (
        "the previous epoch's drift must not leak into the current verdict"
    )


def test_the_epoch_boundary_clears_a_pin_violation(tmp_path: Path) -> None:
    """A declared rebaseline is exactly what makes a changed judge comparable again."""
    from benchlock.model.streams import StreamKind
    from benchlock.model.verdict import AttributionRefusedError

    project = build(tmp_path)
    book = Ledger(project / ".benchlock" / "ledger.jsonl")
    seed_ledger(
        book,
        StreamSpec(
            name="swap",
            seed=3,
            n_runs=30,
            judge_shift=-0.1,
            judge_model_after="claude-sonnet-4-5-20260114",
        ),
        epoch=0,
    )
    with pytest.raises(AttributionRefusedError):
        decide(book.runs(StreamKind.SYSTEM), book.runs(StreamKind.ANCHOR), CONFIG)

    # Same judge change, but each side sits in its own epoch.
    book2 = Ledger(project / ".benchlock" / "ledger2.jsonl")
    seed_ledger(book2, StreamSpec(name="before", seed=3, n_runs=20), epoch=0)
    seed_ledger(
        book2,
        StreamSpec(
            name="after",
            seed=4,
            n_runs=20,
            judge_model="claude-sonnet-4-5-20260114",
        ),
        epoch=1,
    )
    result = decide(book2.runs(StreamKind.SYSTEM), book2.runs(StreamKind.ANCHOR), CONFIG)
    assert result.evidence.epoch == 1
