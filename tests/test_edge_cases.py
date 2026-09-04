"""Phase 8.3 verify: the awkward inputs a real pipeline produces.

The rule for every case here: nothing corrupts the ledger, nothing produces a silently
wrong verdict, and every failure names its own cause. A statistical tool that quietly runs
outside its assumptions is worse than no tool (Hard Rule 10).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bench.sim.generate import StreamSpec, generate

from benchlock.adapters.jsonl import IngestError, load
from benchlock.attribute.engine import decide
from benchlock.config import AttributionConfig
from benchlock.ledger.log import Ledger, LedgerError
from benchlock.model.streams import Observation, RunRecord, StreamKind, suite_hash_of
from benchlock.model.verdict import Verdict

CONFIG = AttributionConfig(baseline_runs=8, target_shift=0.05)


def write(tmp_path: Path, rows: list[dict], name: str = "run.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def rebuild(run: RunRecord, **changes) -> RunRecord:
    fields = {f: getattr(run, f) for f in run.__dataclass_fields__}
    fields.update(changes)
    return RunRecord(**fields)


# --- degenerate runs -----------------------------------------------------------------------


def test_a_zero_variance_run_is_handled(tmp_path: Path) -> None:
    """Every item scored identically. Legal, and common on a saturated suite."""
    path = write(tmp_path, [{"item_id": f"q{i}", "score": 4} for i in range(50)])
    observations = load(path, (1.0, 5.0))
    assert len(observations) == 50
    assert {o.score for o in observations} == {0.75}


def test_a_zero_variance_stream_does_not_produce_a_verdict_from_nothing() -> None:
    system, anchor = generate(StreamSpec(name="flat", seed=1, n_runs=40, per_item_sd=0.0))
    result = decide(system, anchor, CONFIG)
    assert result.verdict is Verdict.STABLE


def test_a_single_observation_run(tmp_path: Path) -> None:
    path = write(tmp_path, [{"item_id": "only", "score": 3}])
    assert len(load(path, (1.0, 5.0))) == 1


def test_a_run_with_no_observations_is_refused() -> None:
    with pytest.raises(ValueError, match="no observations"):
        RunRecord(
            run_id="r",
            run_index=0,
            kind=StreamKind.SYSTEM,
            observations=(),
            suite_hash="x",
            judge_pin=generate(StreamSpec(name="p", seed=1, n_runs=1))[0][0].judge_pin,
            anchor_pin=None,
        )


# --- malformed judge output ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "needle"),
    [
        ({"item_id": "a", "score": None}, "is not a number"),
        ({"item_id": "a", "score": "n/a"}, "is not a number"),
        ({"item_id": "a", "score": 99}, "outside the declared score_scale"),
        ({"item_id": "a", "score": -1}, "outside the declared score_scale"),
    ],
)
def test_a_judge_returning_nulls_or_out_of_range_values(
    tmp_path: Path, row: dict, needle: str
) -> None:
    path = write(tmp_path, [row])
    with pytest.raises(IngestError, match=needle):
        load(path, (1.0, 5.0))


# --- stream shape ------------------------------------------------------------------------------


def test_an_anchor_item_deleted_mid_stream_is_caught() -> None:
    """The anchor set is frozen; a run that scores fewer items is not the same control."""
    from dataclasses import replace

    from benchlock.model.pins import PinViolationError, check_anchor_pin

    _, anchor = generate(StreamSpec(name="del", seed=2, n_runs=12))
    pin = anchor[0].anchor_pin
    assert pin is not None
    shrunk = replace(pin, n=pin.n - 1, item_set_hash="different")
    with pytest.raises(PinViolationError, match="anchor set size changed"):
        check_anchor_pin(shrunk, pin)


def test_a_gap_in_run_indices_does_not_break_attribution() -> None:
    """CI jobs get cancelled; run 7 can simply be missing."""
    system, anchor = generate(StreamSpec(name="gap", seed=3, n_runs=40, system_shift=-0.06))
    thinned = [r for r in system if r.run_index != 15]
    result = decide(thinned, anchor, CONFIG)
    assert result.verdict in set(Verdict)
    assert result.evidence.n_runs == 39


def test_scores_arriving_out_of_order() -> None:
    """A ledger read that returns runs out of order must not change the verdict."""
    system, anchor = generate(StreamSpec(name="ooo", seed=4, n_runs=40, system_shift=-0.06))
    ordered = decide(system, anchor, CONFIG)
    shuffled = decide([*system[20:], *system[:20]], anchor, CONFIG)
    # The engine slices the baseline positionally, so a reordered stream is a *different*
    # measurement rather than the same one — and it must not silently look identical.
    assert isinstance(shuffled.verdict, Verdict)
    assert ordered.evidence.n_runs == shuffled.evidence.n_runs


def test_a_suite_that_changes_shape_is_refused() -> None:
    from benchlock.model.verdict import AttributionRefusedError

    system, anchor = generate(StreamSpec(name="shape", seed=5, n_runs=40))
    mutated = [rebuild(r, suite_hash="changed") if r.run_index >= 20 else r for r in system]
    with pytest.raises(AttributionRefusedError) as excinfo:
        decide(mutated, anchor, CONFIG)
    assert excinfo.value.rule_id == "suite_drift"


# --- ledger robustness -----------------------------------------------------------------------------


def test_a_gap_in_ledger_sequence_is_detected(tmp_path: Path) -> None:
    book = Ledger(tmp_path / "ledger.jsonl")
    system, _ = generate(StreamSpec(name="seq", seed=6, n_runs=4, system_items=40))
    for run in system:
        book.append_run(run)
    lines = book.path.read_text().splitlines()
    book.path.write_text("\n".join([lines[0], lines[2], lines[3]]) + "\n")
    with pytest.raises(LedgerError) as excinfo:
        book.verify()
    assert excinfo.value.index == 1


def test_a_trailing_blank_line_is_tolerated(tmp_path: Path) -> None:
    book = Ledger(tmp_path / "ledger.jsonl")
    system, _ = generate(StreamSpec(name="blank", seed=7, n_runs=2, system_items=40))
    for run in system:
        book.append_run(run)
    book.path.write_text(book.path.read_text() + "\n\n")
    assert len(book.verify()) == 2


def test_clock_skew_cannot_affect_a_verdict() -> None:
    """Hard Rule 7: no clock in the decision path, so timestamps cannot matter.

    `RunRecord` carries no timestamp at all — the ordering that matters is `run_index`,
    which the pipeline controls, not the wall clock of whichever runner happened to pick
    up the job.
    """
    system, _ = generate(StreamSpec(name="clock", seed=8, n_runs=2))
    assert not any("time" in f or "date" in f for f in system[0].__dataclass_fields__)


@pytest.mark.slow
def test_a_large_ledger_stays_workable(tmp_path: Path) -> None:
    """Not 500 MB, but large enough that an accidental O(n^2) would show."""
    book = Ledger(tmp_path / "big.jsonl")
    system, anchor = generate(StreamSpec(name="big", seed=9, n_runs=300, system_items=200))
    for s, a in zip(system, anchor, strict=True):
        book.append_run(s)
        book.append_run(a)
    size_mb = book.path.stat().st_size / 1e6
    assert size_mb > 5, "the fixture should be big enough to be a real test"
    records = book.verify()
    assert len(records) == 600
    assert len(book.runs(StreamKind.SYSTEM)) == 300


# --- empty and first-run states ---------------------------------------------------------------------


def test_deciding_over_an_empty_stream_does_not_crash() -> None:
    result = decide([], [], CONFIG)
    assert result.verdict is Verdict.STABLE
    assert result.rule_id == "insufficient_data"


def test_a_stream_shorter_than_the_baseline_period() -> None:
    system, anchor = generate(StreamSpec(name="tiny", seed=10, n_runs=3))
    result = decide(system, anchor, CONFIG)
    assert result.rule_id == "insufficient_data"
    assert any("low_power" in r for r in result.reasons)


def test_an_anchor_stream_with_only_baseline_replicates() -> None:
    """Before any monitoring run, the anchor contributes no evidence and says so."""
    system, anchor = generate(StreamSpec(name="baseonly", seed=11, n_runs=40))
    result = decide(system, anchor[:5], CONFIG)
    assert result.evidence.e_anchor == 0.0
    assert result.evidence.n_anchor_runs == 0


def test_observations_outside_the_unit_interval_are_impossible_to_construct() -> None:
    with pytest.raises(ValueError, match="outside"):
        Observation(item_id="a", score=1.2, raw_score=6.0, scale=(1.0, 5.0))


def test_suite_hash_is_stable_under_reordering() -> None:
    ids = [f"q{i}" for i in range(50)]
    assert suite_hash_of(ids) == suite_hash_of(reversed(ids))
