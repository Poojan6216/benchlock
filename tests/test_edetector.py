"""Phase 1.4 verify: the bounded-memory detector may only ever be slower than full memory.

The conservative-pruning test is a Hard Rule 4 test. It is marked mandatory, it runs in
its own CI job, and a single counterexample is a build failure — because a pruning rule
that can *create* an alarm turns a memory optimisation into a source of false rollbacks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchlock.stats.betting import Side
from benchlock.stats.confseq import Interval
from benchlock.stats.edetector import (
    EDetector,
    changepoint_log_weight,
    run_detector,
)
from benchlock.stats.eprocess import BaselineNull, MonitorScale, frozen_baseline_null, split_alpha

ALPHA = 0.05
_, ALPHA_M = split_alpha(ALPHA)


def wide_null() -> BaselineNull:
    return BaselineNull(
        interval=Interval(0.45, 0.55), n_baseline=8, alpha_baseline=0.025, point_estimate=0.5
    )


# --- Hard Rule 4: pruning may only delay -------------------------------------------------


def _compare(xs: list[float], max_candidates: int) -> tuple[float, float, int | None, int | None]:
    """(bounded log_e, full log_e, bounded alarm, full alarm) after the same stream."""
    null = wide_null()
    bounded = EDetector(null, ALPHA_M, max_candidates=max_candidates)
    full = EDetector(null, ALPHA_M, max_candidates=None)
    for x in xs:
        bounded.update(x)
        full.update(x)
        # Checked at *every* step, not only at the end: the guarantee is time-uniform.
        assert bounded.log_e <= full.log_e + 1e-9, (
            f"bounded statistic {bounded.log_e} exceeded full memory {full.log_e} at t={full._t}"
        )
    return bounded.log_e, full.log_e, bounded.alarm_time, full.alarm_time


@pytest.mark.mandatory
@settings(max_examples=600, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    xs=st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=40),
    max_candidates=st.integers(min_value=1, max_value=12),
)
def test_pruning_can_only_delay_detection(xs: list[float], max_candidates: int) -> None:
    """Hard Rule 4. A single counterexample is a build failure."""
    b_log_e, f_log_e, b_alarm, f_alarm = _compare(xs, max_candidates)

    assert b_log_e <= f_log_e + 1e-9, "bounded memory produced more evidence than full memory"
    if b_alarm is not None:
        assert f_alarm is not None, "the bounded detector alarmed where full memory did not"
        assert b_alarm >= f_alarm, (
            f"bounded alarm at {b_alarm} preceded full-memory alarm at {f_alarm}"
        )


@pytest.mark.mandatory
@pytest.mark.slow
def test_pruning_property_over_three_thousand_streams() -> None:
    """The spec's 3000-case sweep, over shapes chosen to stress the pruning rule.

    Hypothesis explores adversarially; this covers volume and the specific shapes where
    candidates' contributions cross over — late changes, oscillation, and spikes.
    """
    rng = np.random.default_rng(20260904)
    cases = 0
    violations: list[str] = []

    for i in range(3000):
        horizon = int(rng.integers(5, 45))
        kind = i % 6
        if kind == 0:  # drift-free
            xs = rng.uniform(0.4, 0.6, horizon)
        elif kind == 1:  # late step change, the case pruning is most likely to lose
            cp = max(1, horizon - 5)
            xs = np.concatenate([rng.uniform(0.4, 0.6, cp), rng.uniform(0.85, 1.0, horizon - cp)])
        elif kind == 2:  # early step change
            cp = min(3, horizon - 1)
            xs = np.concatenate([rng.uniform(0.4, 0.6, cp), rng.uniform(0.0, 0.15, horizon - cp)])
        elif kind == 3:  # oscillating, so candidate rankings churn
            xs = np.where(np.arange(horizon) % 2 == 0, 0.95, 0.05).astype(float)
        elif kind == 4:  # single spike
            xs = np.full(horizon, 0.5)
            xs[rng.integers(0, horizon)] = 1.0
        else:  # gradual ramp
            xs = np.clip(np.linspace(0.5, rng.uniform(0.0, 1.0), horizon), 0, 1)

        max_candidates = int(rng.integers(1, 8))
        null = wide_null()
        bounded = EDetector(null, ALPHA_M, max_candidates=max_candidates)
        full = EDetector(null, ALPHA_M, max_candidates=None)
        for t, x in enumerate(xs.tolist()):
            bounded.update(x)
            full.update(x)
            if bounded.log_e > full.log_e + 1e-9:
                violations.append(f"case {i} t={t}: {bounded.log_e} > {full.log_e}")
            if bounded.crossed() and not full.crossed():
                violations.append(f"case {i} t={t}: bounded alarmed where full memory did not")
        if bounded.alarm_time is not None and (
            full.alarm_time is None or bounded.alarm_time < full.alarm_time
        ):
            violations.append(f"case {i}: bounded alarm {bounded.alarm_time} < {full.alarm_time}")
        cases += 1

    assert cases == 3000
    assert not violations, "conservative pruning violated:\n" + "\n".join(violations[:10])


@pytest.mark.mandatory
def test_reweighting_survivors_upward_would_break_the_guarantee() -> None:
    """Negative control: proves the property test can fail.

    If a prune redistributed the dropped mass onto the survivors, the bounded statistic
    could exceed full memory. We do not do that; this asserts the test would notice.
    """
    from benchlock.stats.edetector import ReferenceEDetector

    null = wide_null()
    xs = [0.98] * 12

    bounded = ReferenceEDetector(null, ALPHA_M, max_candidates=2)
    full = ReferenceEDetector(null, ALPHA_M, max_candidates=None)
    bounded.update_many(xs)
    full.update_many(xs)
    assert bounded.log_e <= full.log_e

    # Now simulate the forbidden rule: renormalise the survivors' weights to sum to 1.
    cheating = ReferenceEDetector(null, ALPHA_M, max_candidates=2)
    cheating.update_many(xs)
    survivors = cheating._candidates
    total = math.log(sum(math.exp(c.log_weight) for c in survivors))
    for cand in survivors:
        cand.log_weight -= total  # renormalise upward — the thing Hard Rule 4 forbids
    assert cheating.log_e > full.log_e, "the negative control failed to break the guarantee"

    # And the same rule applied to the vectorised detector breaks it identically.
    fast = EDetector(null, ALPHA_M, max_candidates=2)
    fast.update_many(xs)
    fast_full = EDetector(null, ALPHA_M, max_candidates=None)
    fast_full.update_many(xs)
    assert fast.log_e <= fast_full.log_e
    fast._log_weight[: fast._count] -= math.log(
        float(np.exp(fast._log_weight[: fast._count]).sum())
    )
    fast._log_e_cache = None  # the statistic is cached per update; we just changed inputs
    assert fast.log_e > fast_full.log_e


# --- validity and power --------------------------------------------------------------------


def test_weights_are_a_probability_distribution() -> None:
    total = sum(math.exp(changepoint_log_weight(j)) for j in range(200_000))
    assert total == pytest.approx(1.0, abs=1e-4), "the change-point prior must sum to one"
    with pytest.raises(ValueError, match=">= 0"):
        changepoint_log_weight(-1)


def test_false_alarm_rate_is_bounded_on_drift_free_streams() -> None:
    """The detector inherits Ville's inequality from the weighted sum."""
    rng = np.random.default_rng(11)
    run_mean_sd = 0.08 / math.sqrt(260)
    scale = MonitorScale.from_noise_floor(run_mean_sd)
    null = frozen_baseline_null(run_mean_sd, 5, ALPHA, scale=scale)

    alarms = 0
    trials = 300
    for _ in range(trials):
        detector = EDetector(null, ALPHA_M, max_candidates=64)
        for _ in range(100):
            detector.update(scale.to_unit(float(rng.normal(0.0, run_mean_sd))))
        alarms += int(detector.crossed())
    assert alarms / trials <= ALPHA


def test_a_late_change_is_detected_where_a_fixed_start_process_would_struggle() -> None:
    """The reason for hypothesising a change point at every run."""
    rng = np.random.default_rng(5)
    run_mean_sd = 0.08 / math.sqrt(260)
    scale = MonitorScale.from_noise_floor(run_mean_sd)
    null = frozen_baseline_null(run_mean_sd, 5, ALPHA, scale=scale)

    quiet = [scale.to_unit(float(rng.normal(0.0, run_mean_sd))) for _ in range(60)]
    shifted = [scale.to_unit(float(rng.normal(-0.05, run_mean_sd))) for _ in range(40)]
    detector, trace = run_detector(quiet + shifted, null, ALPHA_M)
    assert detector.crossed(), "a change 60 runs in must still be detected"
    assert detector.alarm_time is not None and detector.alarm_time >= 60
    assert len(trace.log_e) == 100
    # The detector localises the change to roughly where it happened.
    assert detector.best_changepoint is not None
    assert 50 <= detector.best_changepoint <= 75


def test_direction_is_reported() -> None:
    rng = np.random.default_rng(2)
    run_mean_sd = 0.08 / math.sqrt(260)
    scale = MonitorScale.from_noise_floor(run_mean_sd)
    null = frozen_baseline_null(run_mean_sd, 5, ALPHA, scale=scale)
    detector = EDetector(null, ALPHA_M)
    for _ in range(60):
        detector.update(scale.to_unit(float(rng.normal(-0.06, run_mean_sd))))
    assert detector.crossed()
    assert detector.direction is Side.DOWN


def test_memory_stays_bounded() -> None:
    detector = EDetector(wide_null(), ALPHA_M, max_candidates=16)
    detector.update_many([0.5] * 500)
    assert detector.n_candidates <= 16
    assert detector.n_pruned >= 480
    assert detector.state().n_candidates <= 16


def test_state_reports_without_recomputing() -> None:
    detector = EDetector(wide_null(), ALPHA_M)
    detector.update_many([0.9] * 30)
    state = detector.state()
    assert state.t == 30
    assert state.log_e == detector.log_e
    assert state.e_value == pytest.approx(detector.e_value)
    assert state.crossed_at == detector.alarm_time


def test_empty_detector_has_no_evidence() -> None:
    detector = EDetector(wide_null(), ALPHA_M)
    assert detector.log_e == -math.inf
    assert detector.e_value == 0.0
    assert not detector.crossed()
    assert detector.best_changepoint is None
    assert detector.direction is None


@pytest.mark.parametrize(("alpha", "max_candidates"), [(0.0, 8), (1.0, 8), (0.05, 0)])
def test_bad_configuration_is_refused(alpha: float, max_candidates: int) -> None:
    with pytest.raises(ValueError):
        EDetector(wide_null(), alpha, max_candidates=max_candidates)


def test_alarm_time_is_sticky() -> None:
    """Once crossed, the recorded alarm time never moves, even if evidence later falls."""
    rng = np.random.default_rng(8)
    run_mean_sd = 0.08 / math.sqrt(260)
    scale = MonitorScale.from_noise_floor(run_mean_sd)
    null = frozen_baseline_null(run_mean_sd, 5, ALPHA, scale=scale)
    detector = EDetector(null, ALPHA_M)
    detector.update_many([scale.to_unit(-0.08) for _ in range(40)])
    first = detector.alarm_time
    assert first is not None
    detector.update_many([scale.to_unit(float(rng.normal(0, run_mean_sd))) for _ in range(40)])
    assert detector.alarm_time == first


# --- the vectorised detector must be the scalar one, exactly ------------------------------


@pytest.mark.parametrize("strategy_name", ["agrapa", "predmix-eb"])
@pytest.mark.parametrize("max_candidates", [None, 4, 32])
def test_vectorised_detector_matches_the_scalar_reference(
    strategy_name: str, max_candidates: int | None
) -> None:
    """EDetector is a performance rewrite of ReferenceEDetector and nothing more.

    Checked observation by observation, not just at the end, so a divergence cannot hide
    inside a stream and cancel out.
    """
    from benchlock.stats.betting import make_strategy
    from benchlock.stats.edetector import ReferenceEDetector

    rng = np.random.default_rng(20260904)
    for shape in range(6):
        horizon = 40
        if shape == 0:
            xs = rng.uniform(0.4, 0.6, horizon)
        elif shape == 1:
            xs = np.concatenate([rng.uniform(0.45, 0.55, 25), rng.uniform(0.8, 1.0, 15)])
        elif shape == 2:
            xs = np.concatenate([rng.uniform(0.45, 0.55, 5), rng.uniform(0.0, 0.2, 35)])
        elif shape == 3:
            xs = np.where(np.arange(horizon) % 2 == 0, 1.0, 0.0).astype(float)
        elif shape == 4:
            xs = np.full(horizon, 0.5)
        else:
            xs = np.clip(np.linspace(0.5, 0.05, horizon), 0, 1)

        null = wide_null()
        fast = EDetector(
            null, ALPHA_M, max_candidates=max_candidates, strategy=make_strategy(strategy_name)
        )
        slow = ReferenceEDetector(
            null, ALPHA_M, max_candidates=max_candidates, strategy=make_strategy(strategy_name)
        )
        for t, x in enumerate(xs.tolist()):
            fast.update(x)
            slow.update(x)
            assert fast.log_e == pytest.approx(slow.log_e, rel=1e-9, abs=1e-9), (
                f"shape {shape} t={t}: vectorised {fast.log_e} != reference {slow.log_e}"
            )
            assert fast.n_candidates == slow.n_candidates
        assert fast.alarm_time == slow.alarm_time
        assert fast.best_changepoint == slow.best_changepoint
        assert fast.direction == slow.direction


def test_vectorised_detector_refuses_a_strategy_it_cannot_vectorise() -> None:
    from benchlock.stats.betting import FixedBet

    with pytest.raises(ValueError, match="ReferenceEDetector"):
        EDetector(wide_null(), ALPHA_M, strategy=FixedBet(0.3))


def test_vectorised_detector_refuses_out_of_range_observations() -> None:
    detector = EDetector(wide_null(), ALPHA_M)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        detector.update(1.2)


def test_vectorised_detector_is_substantially_faster() -> None:
    """The reason this class exists. Phase 5 decides tens of thousands of streams."""
    import time

    from benchlock.stats.edetector import ReferenceEDetector

    rng = np.random.default_rng(1)
    xs = rng.uniform(0.4, 0.6, 200).tolist()

    start = time.perf_counter()
    EDetector(wide_null(), ALPHA_M, max_candidates=256).update_many(xs)
    fast_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ReferenceEDetector(wide_null(), ALPHA_M, max_candidates=256).update_many(xs)
    slow_seconds = time.perf_counter() - start

    assert fast_seconds < slow_seconds / 5, (
        f"vectorised {fast_seconds:.3f}s vs reference {slow_seconds:.3f}s — "
        "the rewrite is not earning its complexity"
    )
