"""Phase 1.6 verify: the provisioning calculator's monotonicity, and its agreement with reality.

The spec's rule for this file: *if simulation disagrees with the formula, the formula is
wrong; fix the formula, do not adjust the simulation.* It disagreed twice. The formula
originally assumed wealth grows quadratically in the drift, which is only true while the
bet is below its truncation — in practice it is above, and growth is linear. And the
scaling band was set at 10 standard deviations on the reasoning that wider is safer, which
threw away about a third of the detectable-shift resolution. Both were changed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchlock.model.pins import NoiseFloor
from benchlock.stats.edetector import EDetector
from benchlock.stats.eprocess import MonitorScale, frozen_baseline_null, split_alpha
from benchlock.stats.power import (
    DEFAULT_BAND_SDS,
    DETECTION_SAFETY_FACTOR,
    ProvisioningImpossibleError,
    decompose,
    make_plan,
    min_anchor_size,
    min_detectable_shift,
    snapshot_half_width,
)

ALPHA = 0.05
_, ALPHA_M = split_alpha(ALPHA)
RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results"


def floor_at(
    n: int = 200, per_item_sd: float = 0.08, K: int = 5, shared_sd: float = 0.0
) -> NoiseFloor:
    run_mean_sd = math.sqrt(shared_sd**2 + per_item_sd**2 / n)
    return NoiseFloor(per_item_sd=per_item_sd, run_mean_sd=run_mean_sd, replicates=K, n_items=n)


# --- monotonicity -------------------------------------------------------------------------


def test_n_increases_as_the_target_shift_decreases() -> None:
    sizes = [
        min_anchor_size(t, floor_at(), ALPHA, 50, obs_per_run=40)
        for t in (0.10, 0.05, 0.03, 0.02, 0.015)
    ]
    assert sizes == sorted(sizes), f"anchor size must grow as the target shrinks: {sizes}"
    assert sizes[-1] > sizes[0]


def test_n_increases_as_noise_increases() -> None:
    sizes = [
        min_anchor_size(0.02, floor_at(per_item_sd=sd), ALPHA, 50, obs_per_run=40)
        for sd in (0.04, 0.08, 0.12, 0.16)
    ]
    assert sizes == sorted(sizes), f"a noisier judge must need more anchors: {sizes}"
    assert sizes[-1] > sizes[0]


def test_n_increases_as_alpha_decreases() -> None:
    sizes = [
        min_anchor_size(0.02, floor_at(), a, 50, obs_per_run=40) for a in (0.2, 0.1, 0.05, 0.01)
    ]
    assert sizes == sorted(sizes), f"a stricter alpha must need more anchors: {sizes}"
    assert sizes[-1] > sizes[0]


def test_n_increases_as_the_horizon_shortens() -> None:
    sizes = [min_anchor_size(0.02, floor_at(), ALPHA, h, obs_per_run=40) for h in (200, 100, 50)]
    assert sizes == sorted(sizes), f"less time must need more anchors: {sizes}"
    assert sizes[-1] > sizes[0]


def test_a_horizon_too_short_for_any_detection_is_refused_and_says_why() -> None:
    """Below the feasible-horizon floor nothing is provable at any anchor size.

    The old code returned a finite sentinel here that every caller read as a real answer,
    and `decide()` then issued a confident SYSTEM verdict on a pure judge shift. Now the
    calculator refuses, names the floor, and says more anchors cannot help.
    """
    from benchlock.stats.power import min_feasible_horizon

    floor = min_feasible_horizon(ALPHA)
    assert floor > 1
    with pytest.raises(ProvisioningImpossibleError) as excinfo:
        min_anchor_size(0.02, floor_at(), ALPHA, floor - 1, obs_per_run=40)
    assert f"fewer than {floor} runs" in excinfo.value.message
    assert "More anchor items cannot help" in excinfo.value.hint


def test_min_detectable_shift_is_infinite_below_the_feasible_horizon() -> None:
    """Hard Rule 2: an unreachable horizon must read as UNDER_PROVISIONED, never as a number."""
    from benchlock.attribute.race import check_race
    from benchlock.model.verdict import Provisioning
    from benchlock.stats.power import min_feasible_horizon

    short = min_feasible_horizon(ALPHA) - 1
    assert min_detectable_shift(400, floor_at(), ALPHA, short) == float("inf")
    race = check_race(-0.30, floor_at(), 400, short, ALPHA, target_shift=0.05)
    assert race.provisioning is Provisioning.UNDER_PROVISIONED, (
        "a huge observed shift must not launder an anchor process that could not have crossed"
    )


def test_min_detectable_shift_decreases_as_the_anchor_grows() -> None:
    shifts = [min_detectable_shift(n, floor_at(), ALPHA, 50) for n in (25, 50, 100, 400, 1600)]
    assert shifts == sorted(shifts, reverse=True), f"a bigger anchor must see smaller: {shifts}"


def test_min_detectable_shift_decreases_as_the_horizon_lengthens() -> None:
    shifts = [min_detectable_shift(260, floor_at(), ALPHA, h) for h in (10, 25, 50, 200)]
    assert shifts == sorted(shifts, reverse=True)


def test_more_replicates_shrink_the_dead_zone() -> None:
    zones = [snapshot_half_width(0.005, K, ALPHA) for K in (2, 3, 5, 10, 30)]
    assert zones == sorted(zones, reverse=True), f"more replicates must pin the snapshot: {zones}"


def test_the_two_legs_of_the_search_both_bind_somewhere() -> None:
    # Detectability binds when the target is demanding relative to the system stream.
    demanding = min_anchor_size(0.015, floor_at(), ALPHA, 50, obs_per_run=40)
    # The design law binds when the system stream is large and the target is loose.
    race = min_anchor_size(0.10, floor_at(), ALPHA, 50, obs_per_run=500)
    assert demanding > 40, "the detectability leg should dominate for a demanding target"
    assert race >= 500, "the design law should dominate when the system stream is large"


# --- the design law -------------------------------------------------------------------------


def test_the_recommended_anchor_is_never_noisier_than_the_system_stream() -> None:
    """The design law itself: the anchor must not be the slower of the two streams."""
    components = decompose(floor_at())
    for obs_per_run in (20, 40, 200, 1000):
        n = min_anchor_size(0.03, floor_at(), ALPHA, 50, obs_per_run=obs_per_run)
        assert components.run_mean_sd(n) <= components.run_mean_sd(obs_per_run) + 1e-12, (
            f"anchor of {n} is noisier than a system stream of {obs_per_run}"
        )


def test_shared_judge_noise_makes_small_targets_impossible_and_says_so() -> None:
    """An honest refusal beats a big number that would not have worked."""
    correlated = floor_at(shared_sd=0.02)
    assert decompose(correlated).shared_sd > 0.015
    min_anchor_size(0.10, correlated, ALPHA, 50, obs_per_run=40)  # a loose target is fine
    with pytest.raises(ProvisioningImpossibleError) as excinfo:
        min_anchor_size(0.02, correlated, ALPHA, 50, obs_per_run=40)
    assert "does not shrink as the anchor set grows" in excinfo.value.message
    assert "--target-shift" in excinfo.value.hint


def test_decompose_splits_shared_from_independent_noise() -> None:
    components = decompose(floor_at(n=200, per_item_sd=0.08, shared_sd=0.01))
    assert components.shared_sd == pytest.approx(0.01, rel=0.05)
    assert components.per_item_sd == pytest.approx(0.08)
    # A bigger anchor buys down only the independent part.
    assert components.run_mean_sd(1_000_000) == pytest.approx(0.01, abs=1e-4)


def test_decompose_clamps_negative_shared_variance() -> None:
    # A measured run-mean SD below the independent prediction is estimate noise, not
    # evidence of negative variance.
    quiet = NoiseFloor(per_item_sd=0.08, run_mean_sd=0.0001, replicates=5, n_items=200)
    assert decompose(quiet).shared_sd == 0.0


# --- the formula against reality ----------------------------------------------------------------


def _detection_rate(
    anchor_n: int, floor: NoiseFloor, shift: float, horizon: int, trials: int = 200, seed: int = 11
) -> float:
    sigma = decompose(floor).run_mean_sd(anchor_n)
    scale = MonitorScale.from_noise_floor(sigma, band_sds=DEFAULT_BAND_SDS)
    null = frozen_baseline_null(sigma, floor.replicates, ALPHA, scale=scale)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        detector = EDetector(null, ALPHA_M, max_candidates=64)
        for _ in range(horizon):
            detector.update(scale.to_unit(-shift + float(rng.normal(0.0, sigma))))
            if detector.crossed():
                hits += 1
                break
    return hits / trials


@pytest.mark.slow
def test_at_the_recommended_size_the_target_shift_is_actually_detected() -> None:
    """The spec's verify: >= 95% at the recommended n. Measured, and recorded."""
    cases = [
        (0.05, 5, 50, 0.08, 40),
        (0.03, 5, 50, 0.08, 40),
        (0.02, 5, 100, 0.08, 40),
        (0.05, 5, 30, 0.12, 40),
        (0.02, 8, 60, 0.05, 40),
        (0.03, 5, 40, 0.08, 200),
    ]
    rows = []
    for target, K, horizon, per_item_sd, obs_per_run in cases:
        floor = floor_at(per_item_sd=per_item_sd, K=K)
        n = min_anchor_size(target, floor, ALPHA, horizon, obs_per_run=obs_per_run)
        rate = _detection_rate(n, floor, target, horizon)
        rows.append(
            {
                "target_shift": target,
                "replicates": K,
                "horizon": horizon,
                "per_item_sd": per_item_sd,
                "obs_per_run": obs_per_run,
                "recommended_anchor_n": n,
                "detection_rate": rate,
            }
        )
        assert rate >= 0.95, (
            f"target={target} K={K} horizon={horizon}: the recommended n={n} detected the "
            f"target shift only {rate:.2%} of the time. The formula is wrong, not the "
            "simulation"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "provisioning-calibration.json").write_text(
        json.dumps(
            {
                "alpha": ALPHA,
                "safety_factor": DETECTION_SAFETY_FACTOR,
                "band_sds": DEFAULT_BAND_SDS,
                "trials_per_case": 200,
                "cases": rows,
                "note": (
                    "Detection rate at the anchor size `benchlock plan` recommends, for the "
                    "shift it was asked to make detectable. The design law requires the "
                    "anchor process to reach its threshold no later than the system process "
                    "would, so anything below 95% here means the calculator is handing out "
                    "sizes that produce confident, wrong SYSTEM verdicts."
                ),
                "command": "uv run pytest tests/test_power.py -m slow",
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.slow
def test_the_formula_is_conservative_rather_than_optimistic() -> None:
    """`min_detectable_shift` must never claim a shift is visible when it is not.

    Compared against the empirically measured 90%-detection shift: ours should be at or
    above it. Claiming a smaller detectable shift than reality delivers is what produces
    an `INDETERMINATE` that should have been one, so this is the safety-critical direction.
    """
    ratios = []
    for anchor_n, K, horizon, per_item_sd in [
        (260, 5, 50, 0.08),
        (100, 5, 50, 0.08),
        (500, 8, 30, 0.05),
        (260, 10, 100, 0.12),
    ]:
        floor = floor_at(n=anchor_n, per_item_sd=per_item_sd, K=K)
        claimed = min_detectable_shift(anchor_n, floor, ALPHA, horizon)
        empirical = None
        for shift in np.arange(0.002, 0.08, 0.002):
            if _detection_rate(anchor_n, floor, float(shift), horizon, trials=60, seed=3) >= 0.9:
                empirical = float(shift)
                break
        assert empirical is not None, "no shift in the sweep was detectable"
        assert claimed >= empirical * 0.95, (
            f"n={anchor_n} K={K} H={horizon}: formula claims {claimed:.4f} is detectable "
            f"but 90% detection needs {empirical:.4f}. Fix the formula"
        )
        ratios.append(claimed / empirical)
    # Conservative, but not so conservative the tool is useless.
    assert max(ratios) < 2.0, f"the formula is over-conservative by {max(ratios):.2f}x"


# --- plan ------------------------------------------------------------------------------------


def test_make_plan_reports_what_it_can_and_cannot_see() -> None:
    plan = make_plan(0.05, floor_at(), ALPHA, 50, obs_per_run=40)
    assert plan.anchor_n >= 1
    assert plan.achieved_min_detectable_shift <= 0.05
    assert plan.dead_zone > 0.0
    assert plan.dead_zone < plan.achieved_min_detectable_shift
    assert not plan.shared_noise_dominates
    data = plan.to_json()
    assert data["anchor_n"] == plan.anchor_n


def test_make_plan_flags_a_judge_whose_shared_noise_dominates() -> None:
    """The user-facing question is 'will buying more anchor items help?'"""
    # Independent noise: quadrupling the anchor set halves the detectable shift.
    independent = make_plan(0.05, floor_at(), ALPHA, 50, obs_per_run=40)
    assert independent.marginal_gain_at_4x > 0.2
    assert not independent.shared_noise_dominates

    # Shared noise: a bigger anchor set barely moves it, and the plan says so.
    correlated = make_plan(0.05, floor_at(shared_sd=0.012), ALPHA, 50, obs_per_run=40)
    assert correlated.marginal_gain_at_4x < 0.10
    assert correlated.shared_noise_dominates


# --- input validation -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_shift": 0.0}, "target_shift must be positive"),
        ({"obs_per_run": 0}, "obs_per_run must be at least 1"),
    ],
)
def test_bad_inputs_are_refused(kwargs: dict, match: str) -> None:
    call = {"target_shift": 0.05, "obs_per_run": 40}
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        min_anchor_size(
            call["target_shift"], floor_at(), ALPHA, 50, obs_per_run=int(call["obs_per_run"])
        )


def test_min_detectable_shift_refuses_bad_inputs() -> None:
    with pytest.raises(ValueError, match="anchor size must be at least 1"):
        min_detectable_shift(0, floor_at(), ALPHA, 50)
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        min_detectable_shift(100, floor_at(), ALPHA, 0)


def test_snapshot_half_width_needs_two_replicates() -> None:
    with pytest.raises(ValueError, match="at least K=2"):
        snapshot_half_width(0.01, 1, ALPHA)
