"""E-processes for a baseline mean that is *estimated*, not known.

The naive thing is to take the baseline period's sample mean, call it ``mu0``, and monitor
against it. That is wrong, and wrong in the dangerous direction: ``mu_hat`` differs from
the true baseline mean by sampling error, the monitoring process reads that gap as drift,
and the false-alarm rate inflates by an amount nobody measured. The size of that inflation
is measured in ``tests/test_eprocess.py`` and reported in ``docs/statistical-guarantees.md``.

What we do instead:

1. Spend part of the error budget on an anytime-valid **confidence sequence** for the
   baseline mean, giving an interval ``[L, U]`` that covers the truth w.p. >= 1 - alpha_b.
2. Monitor against the **least favourable value in that interval** — the null that is
   hardest to reject given the direction the data have moved. Concretely, the up-leg tests
   ``mu = U`` and the down-leg tests ``mu = L``.

Why that is valid. For any true baseline mean ``mu*`` in ``[L, U]``:

* up-leg, bets ``lambda >= 0``: ``E[1 + lambda (X - U)] = 1 + lambda(mu* - U) <= 1``
* down-leg, bets ``lambda <= 0``: ``E[1 + lambda (X - L)] = 1 + lambda(mu* - L) <= 1``

so each leg is a non-negative supermartingale, their mixture is one, and the max of the
weighted legs is dominated by that mixture. Ville's inequality applies to the whole thing
at level ``alpha_m``, and a union bound with the confidence sequence gives a total
false-alarm probability of at most ``alpha_b + alpha_m = alpha``.

The construction is conservative by design. Conservative is the correct direction for a
tool whose entire claim is that its alarms mean something.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.stats import t as student_t

from benchlock.stats.betting import (
    DEFAULT_TRUNCATION,
    BettingStrategy,
    Side,
    WealthProcess,
)
from benchlock.stats.confseq import Interval, mean_cs

#: Fraction of alpha spent on pinning down the baseline mean. The rest funds monitoring.
DEFAULT_BASELINE_ALPHA_FRACTION = 0.5
#: How close to the [0,1] boundary a null mean may sit. A null at exactly 0 or 1 admits
#: no safe bet, so the interval is nudged inward by this much.
_BOUNDARY_MARGIN = 1e-6


@dataclass(frozen=True, slots=True)
class BaselineNull:
    """The composite null: 'the baseline mean is somewhere in this interval'."""

    interval: Interval
    n_baseline: int
    alpha_baseline: float
    point_estimate: float

    @property
    def width(self) -> float:
        return self.interval.width

    def least_favourable(self, side: Side) -> float:
        """The null value hardest to reject when the data have moved in `side`'s direction.

        Testing against this is what makes the composite null honest: we must reject
        *every* baseline value the data permit, so we test the one most favourable to the
        null hypothesis of no change.
        """
        if side is Side.UP:
            return min(max(self.interval.upper, _BOUNDARY_MARGIN), 1.0 - _BOUNDARY_MARGIN)
        if side is Side.DOWN:
            return min(max(self.interval.lower, _BOUNDARY_MARGIN), 1.0 - _BOUNDARY_MARGIN)
        raise ValueError("least_favourable needs a one-sided Side")

    def to_json(self) -> dict[str, float | int]:
        return {
            "lower": self.interval.lower,
            "upper": self.interval.upper,
            "point_estimate": self.point_estimate,
            "n_baseline": self.n_baseline,
            "alpha_baseline": self.alpha_baseline,
        }


def estimate_baseline_null(
    baseline: Sequence[float],
    alpha: float = 0.05,
    *,
    baseline_alpha_fraction: float = DEFAULT_BASELINE_ALPHA_FRACTION,
    method: str = "hedged",
    breaks: int = 200,
) -> BaselineNull:
    """Anytime-valid interval for the baseline mean, from the baseline period alone."""
    if not baseline:
        raise ValueError(
            "cannot establish a baseline from zero runs; benchlock needs a baseline period "
            "before it can monitor (see `stats.baseline_runs` in benchlock.yaml)"
        )
    if not 0.0 < baseline_alpha_fraction < 1.0:
        raise ValueError(
            f"baseline_alpha_fraction must be in (0, 1), got {baseline_alpha_fraction}"
        )
    alpha_baseline = alpha * baseline_alpha_fraction
    interval = mean_cs(list(baseline), alpha_baseline, method=method, breaks=breaks)[-1]
    return BaselineNull(
        interval=interval,
        n_baseline=len(baseline),
        alpha_baseline=alpha_baseline,
        point_estimate=sum(baseline) / len(baseline),
    )


@dataclass(frozen=True, slots=True)
class MonitorScale:
    """Maps a run statistic onto [0, 1] so a bounded e-process can bet efficiently.

    Betting e-processes are built for observations that use the whole of [0, 1]. Anchor
    run means do not: a stable judge re-scoring a frozen set produces values clustered
    within a few thousandths of the snapshot. Fed raw, the bet is capped at ``c/mu0`` and
    almost no evidence accumulates, however large the drift is *relative to the noise*.

    So we work in units of the measured noise floor: a deviation of ``half_width`` maps to
    the edge of the interval. The transform is affine and fixed in advance, so it changes
    no statistical property — the e-process is still testing a bounded-mean null, just one
    scaled to the resolution the question is actually asked at.

    Deviations beyond the band clip to the boundary. Clipping only ever *reduces* the
    evidence a step contributes, so it cannot manufacture an alarm; a drift large enough
    to clip has already produced far more evidence than it needs.
    """

    center: float  # the value a no-change run maps to, always 0.5
    half_width: float  # a deviation of this size reaches the edge of [0, 1]

    @classmethod
    def from_noise_floor(cls, run_mean_sd: float, band_sds: float = 10.0) -> MonitorScale:
        """Band the stream at `band_sds` run-to-run standard deviations.

        Ten is wide enough that ordinary noise never approaches the edge, and narrow
        enough that a drift of a few noise units is a large move in scaled space.
        """
        if run_mean_sd <= 0.0:
            raise ValueError(
                f"run_mean_sd must be positive to scale a stream, got {run_mean_sd}. "
                "Measure the noise floor with `benchlock baseline` before monitoring"
            )
        return cls(center=0.5, half_width=band_sds * run_mean_sd)

    def to_unit(self, deviation: float) -> float:
        """Scaled deviation, clipped into [0, 1]."""
        return min(max(self.center + deviation / (2.0 * self.half_width), 0.0), 1.0)

    def to_raw_shift(self, unit_shift: float) -> float:
        """Invert the scaling for a *shift*, so intervals can be reported in real units."""
        return unit_shift * 2.0 * self.half_width

    def raw_interval(self, unit: Interval) -> Interval:
        """Map a unit-space interval on the mean back to a raw-space interval on the shift."""
        return Interval(
            self.to_raw_shift(unit.lower - self.center),
            self.to_raw_shift(unit.upper - self.center),
        )


def frozen_baseline_null(
    run_mean_sd: float,
    replicates: int,
    alpha: float = 0.05,
    *,
    baseline_alpha_fraction: float = DEFAULT_BASELINE_ALPHA_FRACTION,
    center: float = 0.5,
    scale: MonitorScale | None = None,
) -> BaselineNull:
    """The null for a stream monitored against a **frozen snapshot** rather than a period.

    This is the anchor stream in ``frozen-self`` mode, and it is the reason the anchor set
    is worth having. Its null mean is not estimated from a handful of runs — it *is* the
    frozen baseline, so the only uncertainty is how precisely that snapshot was measured:
    ``run_mean_sd / sqrt(K)`` for K replicates.

    That distinction matters enormously. An estimated baseline over 8 runs yields a
    composite null roughly 0.6 wide, because no anytime-valid method can pin a bounded
    mean tightly from 8 observations. A frozen snapshot measured with K=5 replicates
    yields one narrower by two orders of magnitude. The anchor is not just a control
    group; it is the only stream whose null is *known*.

    Observations are expected as deviations from the snapshot, already mapped through
    ``scale`` so that the interval below is in the same units the e-process will see.
    """
    if run_mean_sd < 0.0:
        raise ValueError(f"run_mean_sd must be non-negative, got {run_mean_sd}")
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    # The snapshot mean is estimated from K replicates, so bound it with a Student-t
    # quantile on K-1 degrees of freedom rather than a hand-picked multiple of the SE.
    # With K small this is appreciably wider than a normal quantile, which is correct.
    alpha_baseline = alpha * baseline_alpha_fraction
    standard_error = run_mean_sd / math.sqrt(replicates)
    if replicates > 1:
        quantile = float(student_t.ppf(1.0 - alpha_baseline / 2.0, df=replicates - 1))
    else:
        # A single replicate cannot bound its own error; refuse to pretend otherwise.
        raise ValueError(
            "a frozen baseline needs at least K=2 replicates to bound the snapshot's own "
            "measurement error; set `anchor.noise_replicates` to 2 or more"
        )
    half = quantile * standard_error
    if scale is not None:
        half = half / (2.0 * scale.half_width)
    interval = Interval(max(0.0, center - half), min(1.0, center + half))
    return BaselineNull(
        interval=interval,
        n_baseline=replicates,
        alpha_baseline=alpha_baseline,
        point_estimate=center,
    )


class CompositeNullEProcess:
    """Monitors a stream against a baseline mean known only up to an interval.

    The value reported is an **e-value**: under the null its expectation is at most 1 at
    every stopping time, so ``e >= 1/alpha`` is a rejection whose error probability is
    bounded no matter how often it is inspected.
    """

    __slots__ = ("_theta", "alpha_monitor", "down", "null", "up")

    def __init__(
        self,
        null: BaselineNull,
        alpha_monitor: float,
        *,
        strategy: BettingStrategy | None = None,
        c: float = DEFAULT_TRUNCATION,
        theta: float = 0.5,
    ) -> None:
        self.null = null
        self.alpha_monitor = alpha_monitor
        self._theta = theta
        # Each leg tests the null value least favourable to detecting its own direction.
        self.up = WealthProcess(
            null.least_favourable(Side.UP), side=Side.UP, strategy=strategy, c=c
        )
        self.down = WealthProcess(
            null.least_favourable(Side.DOWN), side=Side.DOWN, strategy=strategy, c=c
        )

    @property
    def t(self) -> int:
        return self.up.t

    def update(self, x: float) -> float:
        self.up.update(x)
        self.down.update(x)
        return self.log_e

    def update_many(self, xs: Sequence[float]) -> float:
        for x in xs:
            self.update(x)
        return self.log_e

    @property
    def log_e(self) -> float:
        """max of the weighted legs, in log space. Dominated by their mixture, so valid."""
        return max(
            math.log(self._theta) + self.up.log_wealth,
            math.log(1.0 - self._theta) + self.down.log_wealth,
        )

    @property
    def e_value(self) -> float:
        try:
            return math.exp(self.log_e)
        except OverflowError:  # pragma: no cover
            return math.inf

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha_monitor

    def crossed(self) -> bool:
        return self.log_e >= math.log(self.threshold)

    @property
    def direction(self) -> Side:
        """Which leg holds the evidence. Reported in the verdict, never used to decide it."""
        return Side.UP if self.up.log_wealth >= self.down.log_wealth else Side.DOWN


class PointNullEProcess:
    """The naive version: monitor against the baseline *point estimate*.

    **This is not valid and is never used to produce a verdict.** It exists so Phase 1.2
    can measure how much false-alarm rate the shortcut buys, and so that number can be
    published rather than asserted. Hard Rule 3 keeps it out of the decision path.
    """

    __slots__ = ("alpha_monitor", "down", "up")

    def __init__(
        self,
        mu_hat: float,
        alpha_monitor: float,
        *,
        strategy: BettingStrategy | None = None,
        c: float = DEFAULT_TRUNCATION,
    ) -> None:
        mu0 = min(max(mu_hat, _BOUNDARY_MARGIN), 1.0 - _BOUNDARY_MARGIN)
        self.alpha_monitor = alpha_monitor
        self.up = WealthProcess(mu0, side=Side.UP, strategy=strategy, c=c)
        self.down = WealthProcess(mu0, side=Side.DOWN, strategy=strategy, c=c)

    def update(self, x: float) -> float:
        self.up.update(x)
        self.down.update(x)
        return self.log_e

    def update_many(self, xs: Sequence[float]) -> float:
        for x in xs:
            self.update(x)
        return self.log_e

    @property
    def log_e(self) -> float:
        return max(math.log(0.5) + self.up.log_wealth, math.log(0.5) + self.down.log_wealth)

    @property
    def e_value(self) -> float:
        try:
            return math.exp(self.log_e)
        except OverflowError:  # pragma: no cover
            return math.inf

    def crossed(self) -> bool:
        return self.log_e >= math.log(1.0 / self.alpha_monitor)


def split_alpha(
    alpha: float, baseline_alpha_fraction: float = DEFAULT_BASELINE_ALPHA_FRACTION
) -> tuple[float, float]:
    """(alpha_baseline, alpha_monitor). They sum to alpha, by the union bound."""
    alpha_baseline = alpha * baseline_alpha_fraction
    return alpha_baseline, alpha - alpha_baseline
