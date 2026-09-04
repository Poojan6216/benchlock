"""Phase 1.5 verify: characterise a judge with known noise, and prove the floor earns its place."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchlock.anchor.noisefloor import (
    MIN_SD,
    NoiseFloorError,
    estimate_noise_floor,
    independence_ratio,
)
from benchlock.model.pins import NoiseFloor
from benchlock.stats.edetector import EDetector
from benchlock.stats.eprocess import MonitorScale, frozen_baseline_null, split_alpha

ALPHA = 0.05
_, ALPHA_M = split_alpha(ALPHA)


def synthetic_judge(
    rng: np.random.Generator,
    n_items: int,
    replicates: int,
    per_item_sd: float,
    *,
    shared_sd: float = 0.0,
    truth: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """A judge with known, injected nondeterminism.

    `per_item_sd` is independent per-item wobble; `shared_sd` is a run-wide offset that
    moves every item together, which is how a real provider-side change looks.
    """
    base = truth if truth is not None else rng.uniform(0.3, 0.9, n_items)
    out = []
    for _ in range(replicates):
        shared = rng.normal(0.0, shared_sd) if shared_sd > 0 else 0.0
        scores = np.clip(base + rng.normal(0.0, per_item_sd, n_items) + shared, 0.0, 1.0)
        out.append({f"a{i}": float(s) for i, s in enumerate(scores)})
    return out


# --- characterisation ----------------------------------------------------------------------


@pytest.mark.parametrize("true_sd", [0.01, 0.05, 0.12])
def test_known_per_item_noise_is_recovered(true_sd: float) -> None:
    rng = np.random.default_rng(20260904)
    # Centre the truth away from the bounds so clipping does not bias the estimate.
    truth = rng.uniform(0.35, 0.65, 400)
    report = estimate_noise_floor(synthetic_judge(rng, 400, 8, true_sd, truth=truth))
    assert report.floor.per_item_sd == pytest.approx(true_sd, rel=0.15)
    assert report.floor.replicates == 8
    assert report.floor.n_items == 400
    assert not report.degenerate


def test_run_mean_sd_matches_the_independent_prediction_when_noise_is_independent() -> None:
    rng = np.random.default_rng(7)
    n = 500
    truth = rng.uniform(0.35, 0.65, n)
    report = estimate_noise_floor(synthetic_judge(rng, n, 30, 0.08, truth=truth))
    predicted = 0.08 / math.sqrt(n)
    assert report.floor.run_mean_sd == pytest.approx(predicted, rel=0.4)
    assert independence_ratio(report.floor) == pytest.approx(1.0, abs=0.5)


def test_correlated_judge_noise_is_caught_by_measuring_run_means_directly() -> None:
    """The reason run_mean_sd is measured, not derived.

    A judge that shifts every item together between calls has a run-mean SD far larger
    than `per_item_sd/sqrt(n)`. Deriving it would understate the floor and produce a
    detector that false-alarms.
    """
    rng = np.random.default_rng(3)
    n = 400
    truth = rng.uniform(0.35, 0.65, n)
    report = estimate_noise_floor(synthetic_judge(rng, n, 30, 0.05, shared_sd=0.02, truth=truth))
    derived = report.floor.per_item_sd / math.sqrt(n)
    assert report.floor.run_mean_sd > 5 * derived
    assert independence_ratio(report.floor) > 3.0


def test_self_disagreement_diagnostics() -> None:
    rng = np.random.default_rng(1)
    report = estimate_noise_floor(synthetic_judge(rng, 200, 5, 0.06))
    assert 0.0 < report.mean_abs_pairwise_diff < 0.2
    assert report.max_item_disagreement > report.mean_abs_pairwise_diff
    assert report.unstable_items > 150
    assert report.exact_agreement_rate < 0.05  # continuous scores rarely tie exactly


def test_a_deterministic_judge_is_reported_as_degenerate_not_as_zero_noise() -> None:
    """K identical scorings do not prove zero noise; they prove K was too small to see it."""
    identical = [{"a": 0.5, "b": 0.7} for _ in range(5)]
    report = estimate_noise_floor(identical)
    assert report.degenerate
    assert report.floor.run_mean_sd == MIN_SD
    assert report.floor.per_item_sd == MIN_SD
    assert report.exact_agreement_rate == 1.0
    assert any("identical scores" in w for w in report.warnings())


def test_small_k_is_warned_about() -> None:
    rng = np.random.default_rng(2)
    report = estimate_noise_floor(synthetic_judge(rng, 50, 2, 0.05))
    assert any("degree of freedom" in w for w in report.warnings())


# --- the test that proves the component earns its place -----------------------------------------


@pytest.mark.slow
def test_a_zero_noise_floor_false_alarms_and_the_measured_one_does_not() -> None:
    """Phase 1.5's headline check: the floor earns its place, measured.

    What the floor actually protects against is **error in the frozen snapshot**. The
    baseline scores are themselves one measurement of a noisy judge, so they sit an
    offset ``eps`` away from the judge's true mean. Every later run is then compared
    against a slightly wrong reference, and that offset is *persistent* — it never
    averages away, and a detector reads it as permanent drift.

    Sizing the null to cover ``eps ~ sd/sqrt(K)`` is exactly what stops that. Here both
    arms see identical drift-free streams and use identical scaling; the only difference
    is whether the null was sized against the measured floor or against a judge assumed
    to be quiet.
    """
    rng = np.random.default_rng(20260904)
    n_items, K = 260, 5
    true_per_item_sd = 0.08
    truth = rng.uniform(0.35, 0.65, n_items)

    report = estimate_noise_floor(synthetic_judge(rng, n_items, K, true_per_item_sd, truth=truth))
    measured_sd = report.floor.run_mean_sd
    assert measured_sd > 10 * MIN_SD, "the synthetic judge must actually be noisy"

    # Scaling is a change of variable and is held fixed, so the two arms differ in
    # exactly one thing: how wide the null is.
    scale = MonitorScale.from_noise_floor(measured_sd)
    optimistic_sd = measured_sd / 50.0  # "the judge is basically deterministic"

    horizon, trials = 300, 100

    def false_alarm_rate(null_sd: float) -> float:
        alarms = 0
        local = np.random.default_rng(99)
        null = frozen_baseline_null(null_sd, K, ALPHA, scale=scale)
        for _ in range(trials):
            # The snapshot was measured once from K replicates, so it is off by this much.
            snapshot_error = float(local.normal(0.0, measured_sd / math.sqrt(K)))
            detector = EDetector(null, ALPHA_M, max_candidates=32)
            for _ in range(horizon):
                # Drift-free: the judge has not moved. The only systematic component is
                # the snapshot being slightly wrong — and that component never averages
                # away, so a null too narrow to cover it accumulates it as evidence.
                deviation = -snapshot_error + float(local.normal(0.0, measured_sd))
                detector.update(scale.to_unit(deviation))
            alarms += int(detector.crossed())
        return alarms / trials

    optimistic_rate = false_alarm_rate(optimistic_sd)
    measured_rate = false_alarm_rate(measured_sd)

    summary = {
        "anchor_items": n_items,
        "replicates": K,
        "true_per_item_sd": true_per_item_sd,
        "measured_run_mean_sd": measured_sd,
        "horizon": horizon,
        "trials": trials,
        "alpha": ALPHA,
        "measured_floor_false_alarm_rate": measured_rate,
        "assumed_quiet_judge_false_alarm_rate": optimistic_rate,
        "note": (
            "Both arms monitor identical drift-free streams with identical scaling. They "
            "differ only in how wide the null is: sized against the measured noise floor, "
            "or against a judge assumed 50x quieter than it is. The frozen snapshot is "
            "itself one noisy measurement, so it sits a persistent offset away from the "
            "judge's true mean, and a null too narrow to cover that offset reads it as "
            "drift that never goes away."
        ),
        "command": "uv run pytest tests/test_noisefloor.py -k zero_noise_floor",
    }
    results = Path(__file__).resolve().parent.parent / "bench" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "noise-floor-earns-its-place.json").write_text(json.dumps(summary, indent=2) + "\n")

    assert measured_rate <= ALPHA, f"the measured floor should hold alpha, got {measured_rate:.3f}"
    assert optimistic_rate >= 0.2, (
        "a detector that assumes the judge is quiet should read snapshot error as drift, "
        f"got {optimistic_rate:.3f}"
    )
    assert optimistic_rate > 5 * max(measured_rate, 0.01)


# --- input validation -----------------------------------------------------------------------------


def test_fewer_than_two_replicates_is_refused() -> None:
    with pytest.raises(NoiseFloorError, match="at least 2 replicates"):
        estimate_noise_floor([{"a": 0.5}])


def test_mismatched_item_sets_are_refused() -> None:
    with pytest.raises(NoiseFloorError, match="different item set"):
        estimate_noise_floor([{"a": 0.5, "b": 0.5}, {"a": 0.5, "c": 0.5}])


def test_empty_items_are_refused() -> None:
    with pytest.raises(NoiseFloorError, match="no items"):
        estimate_noise_floor([{}, {}])


def test_noise_floor_validates_its_own_fields() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NoiseFloor(per_item_sd=-0.1, run_mean_sd=0.01, replicates=5, n_items=10)
    with pytest.raises(ValueError, match="K=2"):
        NoiseFloor(per_item_sd=0.1, run_mean_sd=0.01, replicates=1, n_items=10)
    with pytest.raises(ValueError, match="at least one item"):
        NoiseFloor(per_item_sd=0.1, run_mean_sd=0.01, replicates=5, n_items=0)


def test_report_serialises() -> None:
    rng = np.random.default_rng(4)
    report = estimate_noise_floor(synthetic_judge(rng, 30, 4, 0.05))
    data = report.to_json()
    assert set(data) >= {"per_item_sd", "run_mean_sd", "replicates", "n_items", "degenerate"}
