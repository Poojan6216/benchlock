"""Phase 1.2 verify: the composite-null e-process holds its false-alarm rate.

The headline simulation compares it against the naive point-estimate version, which is
what a reasonable person would build first. The difference between them is the cost of
not knowing the baseline mean, and it is large enough to matter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchlock.stats.betting import Side
from benchlock.stats.confseq import Interval
from benchlock.stats.eprocess import (
    BaselineNull,
    CompositeNullEProcess,
    MonitorScale,
    PointNullEProcess,
    estimate_baseline_null,
    frozen_baseline_null,
    split_alpha,
)

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results"


def run_stream(
    rng: np.random.Generator,
    mu: float,
    n_baseline: int,
    horizon: int,
    alpha: float,
    *,
    obs_per_run: int = 40,
    drift: float = 0.0,
) -> tuple[bool, bool]:
    """One (baseline, monitoring) stream. Returns (composite alarmed, naive alarmed)."""
    baseline = [float(rng.binomial(obs_per_run, mu) / obs_per_run) for _ in range(n_baseline)]
    _alpha_b, alpha_m = split_alpha(alpha)
    null = estimate_baseline_null(baseline, alpha, breaks=200)

    composite = CompositeNullEProcess(null, alpha_m)
    naive = PointNullEProcess(null.point_estimate, alpha)

    composite_alarm = False
    naive_alarm = False
    for _ in range(horizon):
        x = float(rng.binomial(obs_per_run, min(max(mu + drift, 0.0), 1.0)) / obs_per_run)
        composite.update(x)
        naive.update(x)
        composite_alarm = composite_alarm or composite.crossed()
        naive_alarm = naive_alarm or naive.crossed()
    return composite_alarm, naive_alarm


@pytest.mark.slow
def test_false_alarm_rate_is_bounded_and_the_naive_version_is_not() -> None:
    """2000 drift-free streams, horizon 300, peeked at every step.

    The composite version must hold alpha. The naive one is measured, not asserted to
    pass — the number it produces is the point of the experiment.
    """
    alpha, n_streams, horizon, n_baseline = 0.05, 2000, 300, 12
    rng = np.random.default_rng(20260904)

    composite_alarms = 0
    naive_alarms = 0
    for _ in range(n_streams):
        mu = float(rng.uniform(0.3, 0.7))
        c_alarm, n_alarm = run_stream(rng, mu, n_baseline, horizon, alpha)
        composite_alarms += int(c_alarm)
        naive_alarms += int(n_alarm)

    composite_rate = composite_alarms / n_streams
    naive_rate = naive_alarms / n_streams

    summary = {
        "streams": n_streams,
        "horizon": horizon,
        "baseline_runs": n_baseline,
        "obs_per_run": 40,
        "alpha": alpha,
        "composite_null_false_alarm_rate": composite_rate,
        "point_estimate_false_alarm_rate": naive_rate,
        "composite_within_alpha": composite_rate <= alpha,
        "point_estimate_over_alpha_factor": round(naive_rate / alpha, 2),
        "note": (
            "Both processes monitor the same drift-free streams and are inspected at "
            "every one of the 300 runs. The point-estimate version is the natural first "
            "implementation: take the baseline sample mean and monitor against it. It is "
            "invalid, and this is the size of the error."
        ),
        "command": "uv run pytest tests/test_eprocess.py -m slow",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eprocess-false-alarm.json").write_text(json.dumps(summary, indent=2) + "\n")

    assert composite_rate <= alpha, (
        f"composite-null e-process false-alarm rate {composite_rate:.4f} exceeds alpha={alpha}"
    )
    # Not an assertion about the naive version being bad — a record of what it measured.
    assert naive_rate >= composite_rate


def test_composite_null_holds_alpha_on_a_short_horizon() -> None:
    """A faster version of the headline simulation, run on every commit."""
    alpha, n_streams, horizon = 0.05, 300, 60
    rng = np.random.default_rng(4242)
    alarms = 0
    for _ in range(n_streams):
        mu = float(rng.uniform(0.35, 0.65))
        alarmed, _ = run_stream(rng, mu, 10, horizon, alpha)
        alarms += int(alarmed)
    assert alarms / n_streams <= alpha


def test_a_real_shift_is_still_detected_with_an_estimated_baseline() -> None:
    """Conservative must not mean blind — but an estimated baseline is expensive.

    With 50 baseline runs the composite null is roughly +/-0.08 wide, so only shifts
    clearly outside it are detectable. That is a real limitation of estimating a bounded
    mean from a handful of runs, it is why `benchlock plan` exists, and it is measured
    rather than hidden.
    """
    rng = np.random.default_rng(7)
    detected = 0
    trials = 30
    for _ in range(trials):
        alarmed, _ = run_stream(rng, 0.5, 50, 120, 0.05, drift=-0.25)
        detected += int(alarmed)
    assert detected / trials >= 0.9, "a 0.25 shift must be caught with a 50-run baseline"


def test_the_frozen_snapshot_null_is_orders_of_magnitude_tighter() -> None:
    """Why the anchor set earns its cost.

    An estimated baseline over 8 runs cannot pin a bounded mean to better than a few
    tenths. A frozen snapshot measured with K replicates pins it to a few thousandths,
    because its null is *known* rather than inferred. This gap is the entire reason the
    anchor stream can attribute a judge shift that the system stream cannot.
    """
    rng = np.random.default_rng(3)
    estimated = estimate_baseline_null(
        [float(rng.binomial(40, 0.5) / 40) for _ in range(8)], 0.05, breaks=200
    )
    run_mean_sd = 0.08 / math.sqrt(260)  # 260 anchor items, per-item judge noise 0.08
    scale = MonitorScale.from_noise_floor(run_mean_sd)
    frozen = frozen_baseline_null(run_mean_sd, 5, 0.05, scale=scale)
    frozen_raw_width = scale.to_raw_shift(frozen.width)

    assert frozen_raw_width < estimated.width / 10.0, (
        f"frozen null width {frozen_raw_width:.5f} should be far tighter than the "
        f"estimated {estimated.width:.5f}"
    )


@pytest.mark.parametrize(
    ("anchor_n", "shift", "should_detect"),
    [
        (260, 0.02, True),  # a well-provisioned anchor catches a small judge shift
        (260, 0.05, True),
        (40, 0.02, False),  # an under-provisioned one cannot — this is Demo 3
        (40, 0.10, True),
    ],
)
def test_frozen_anchor_detection_scales_with_anchor_size(
    anchor_n: int, shift: float, should_detect: bool
) -> None:
    """The provisioning story, measured: a bigger anchor set sees smaller judge shifts."""
    rng = np.random.default_rng(20260904 + anchor_n)
    run_mean_sd = 0.08 / math.sqrt(anchor_n)
    scale = MonitorScale.from_noise_floor(run_mean_sd)
    null = frozen_baseline_null(run_mean_sd, 5, 0.05, scale=scale)
    _, alpha_m = split_alpha(0.05)

    hits = 0
    trials = 20
    for _ in range(trials):
        process = CompositeNullEProcess(null, alpha_m)
        for _ in range(100):
            process.update(scale.to_unit(float(rng.normal(-shift, run_mean_sd))))
            if process.crossed():
                hits += 1
                break
    rate = hits / trials
    if should_detect:
        assert rate >= 0.9, f"anchor n={anchor_n} failed to detect a {shift} shift ({rate:.2f})"
    else:
        assert rate <= 0.1, f"anchor n={anchor_n} claimed to detect a {shift} shift ({rate:.2f})"


def test_frozen_null_with_two_replicates_refuses_to_claim_precision() -> None:
    # A t quantile on 1 degree of freedom is enormous, so the null covers everything and
    # nothing can be rejected. That is the honest answer, not a bug.
    run_mean_sd = 0.08 / math.sqrt(260)
    scale = MonitorScale.from_noise_floor(run_mean_sd)
    null = frozen_baseline_null(run_mean_sd, 2, 0.05, scale=scale)
    assert null.interval.lower == 0.0
    assert null.interval.upper == 1.0


def test_frozen_null_needs_at_least_two_replicates() -> None:
    with pytest.raises(ValueError, match="at least K=2 replicates"):
        frozen_baseline_null(0.01, 1, 0.05)


def test_monitor_scale_round_trips_and_clips() -> None:
    scale = MonitorScale.from_noise_floor(0.005, band_sds=10.0)
    assert scale.half_width == pytest.approx(0.05)
    assert scale.to_unit(0.0) == pytest.approx(0.5)
    assert scale.to_unit(0.05) == pytest.approx(1.0)
    assert scale.to_unit(-0.05) == pytest.approx(0.0)
    # Beyond the band, clip rather than leave [0,1]; clipping only ever removes evidence.
    assert scale.to_unit(0.5) == 1.0
    assert scale.to_unit(-0.5) == 0.0
    assert scale.to_raw_shift(scale.to_unit(0.02) - 0.5) == pytest.approx(0.02)


def test_monitor_scale_refuses_a_zero_noise_floor() -> None:
    with pytest.raises(ValueError, match="run_mean_sd must be positive"):
        MonitorScale.from_noise_floor(0.0)


# --- the construction itself -------------------------------------------------------------


def test_least_favourable_values_are_the_interval_endpoints() -> None:
    null = BaselineNull(Interval(0.4, 0.6), n_baseline=10, alpha_baseline=0.025, point_estimate=0.5)
    # Detecting an increase means ruling out every baseline the data permit, hardest of
    # which is the largest one.
    assert null.least_favourable(Side.UP) == pytest.approx(0.6)
    assert null.least_favourable(Side.DOWN) == pytest.approx(0.4)


def test_least_favourable_is_nudged_off_the_boundary() -> None:
    null = BaselineNull(Interval(0.0, 1.0), n_baseline=3, alpha_baseline=0.025, point_estimate=0.5)
    assert 0.0 < null.least_favourable(Side.DOWN) < 1.0
    assert 0.0 < null.least_favourable(Side.UP) < 1.0


def test_two_sided_side_is_rejected() -> None:
    null = BaselineNull(Interval(0.4, 0.6), n_baseline=10, alpha_baseline=0.025, point_estimate=0.5)
    with pytest.raises(ValueError, match="one-sided"):
        null.least_favourable(Side.TWO_SIDED)


def test_the_composite_null_is_wider_than_a_point_and_therefore_slower() -> None:
    """The price of honesty, made explicit: a wider null needs more evidence."""
    rng = np.random.default_rng(3)
    baseline = [float(rng.binomial(40, 0.6) / 40) for _ in range(10)]
    null = estimate_baseline_null(baseline, 0.05, breaks=200)
    assert null.width > 0.0

    monitoring = [float(rng.binomial(40, 0.45) / 40) for _ in range(60)]
    composite = CompositeNullEProcess(null, 0.025)
    naive = PointNullEProcess(null.point_estimate, 0.05)
    composite.update_many(monitoring)
    naive.update_many(monitoring)
    assert composite.log_e <= naive.log_e, "the composite null cannot beat a point null"


def test_split_alpha_sums_to_alpha() -> None:
    a_b, a_m = split_alpha(0.05)
    assert a_b + a_m == pytest.approx(0.05)
    assert a_b > 0 and a_m > 0


def test_empty_baseline_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot establish a baseline"):
        estimate_baseline_null([], 0.05)


def test_bad_alpha_fraction_is_refused() -> None:
    with pytest.raises(ValueError, match="baseline_alpha_fraction"):
        estimate_baseline_null([0.5], 0.05, baseline_alpha_fraction=1.0)


def test_direction_reports_where_the_evidence_is() -> None:
    null = BaselineNull(
        Interval(0.45, 0.55), n_baseline=10, alpha_baseline=0.025, point_estimate=0.5
    )
    down = CompositeNullEProcess(null, 0.025)
    down.update_many([0.05] * 30)
    assert down.direction is Side.DOWN
    up = CompositeNullEProcess(null, 0.025)
    up.update_many([0.95] * 30)
    assert up.direction is Side.UP


def test_e_value_and_threshold_are_consistent() -> None:
    null = BaselineNull(
        Interval(0.45, 0.55), n_baseline=10, alpha_baseline=0.025, point_estimate=0.5
    )
    process = CompositeNullEProcess(null, 0.025)
    assert process.threshold == pytest.approx(40.0)
    assert process.e_value == pytest.approx(0.5)  # max(0.5*1, 0.5*1) before any data
    assert not process.crossed()
    process.update_many([0.99] * 200)
    assert process.crossed()
    assert process.e_value >= process.threshold
    assert math.isfinite(process.log_e)


def test_target_sized_band_takes_the_larger_of_target_and_noise_floor() -> None:
    """The band expresses the scale of change you care about, floored by the noise."""
    # A demanding target, well above the noise floor: the target sets the band.
    wide = MonitorScale.for_target(0.05, 0.005)
    assert wide.half_width == pytest.approx(0.05)
    assert wide.to_unit(-0.05) == pytest.approx(0.0), "the target should reach the edge"

    # A target below four standard deviations: the noise floor sets the band instead,
    # because a band that ordinary noise clips against destroys the signal's shape.
    narrow = MonitorScale.for_target(0.001, 0.005)
    assert narrow.half_width == pytest.approx(0.02)


def test_target_sized_band_refuses_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="target_shift must be positive"):
        MonitorScale.for_target(0.0, 0.005)
    with pytest.raises(ValueError, match="run_mean_sd must be positive"):
        MonitorScale.for_target(0.05, 0.0)
