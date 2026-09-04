"""Time-uniform confidence sequences for a bounded mean.

A confidence sequence is a sequence of intervals ``(L_t, U_t)`` with

    P( for all t : mu in (L_t, U_t) )  >=  1 - alpha

Note the quantifier: the coverage holds *simultaneously over all t*, not at each t
separately. That is what makes it safe to read the interval after every CI run, which is
the only way anyone actually uses one.

Two constructions, both from Waudby-Smith & Ramdas, *Estimating means of bounded random
variables by betting*:

* :func:`hedged_cs` inverts the hedged capital martingale — for each candidate mean, ask
  whether a bettor testing that null would have got rich. Tight, and the default.
* :func:`empirical_bernstein_cs` is a closed-form predictable-mixture bound. Cheaper, a
  little wider, and useful as a cross-check.

Both are implemented against the same reference construction as the `confseq` package, so
the Phase 1.3 differential test is a like-for-like comparison rather than a coincidence.
We implement rather than import because the monitoring layer needs composite nulls and a
change-point-robust detector, neither of which is a `confseq` call — but a construction
validated against nothing is not a construction anyone should trust.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DEFAULT_TRUNCATION = 0.5
DEFAULT_PRIOR_MEAN = 0.5
DEFAULT_PRIOR_VARIANCE = 0.25
DEFAULT_FAKE_OBS = 1.0
#: Grid resolution when inverting the betting martingale. The reference uses 1000.
DEFAULT_BREAKS = 1000
#: Tolerance for the log-space threshold comparison. Ten orders of magnitude below any
#: statistically meaningful difference; exists only to break exact ties outward.
_LOG_TIE_SLACK = 1e-10


@dataclass(frozen=True, slots=True)
class Interval:
    """A closed interval on the mean. Always within [0, 1] for bounded scores."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.upper < self.lower:
            raise ValueError(f"interval is inverted: [{self.lower}, {self.upper}]")

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def excludes_zero_shift(self) -> bool:
        """For a *shift* interval: does it rule out 'no change'?"""
        return not self.contains(0.0)

    def intersect(self, other: Interval) -> Interval:
        return Interval(max(self.lower, other.lower), min(self.upper, other.upper))

    def shifted(self, by: float) -> Interval:
        return Interval(self.lower + by, self.upper + by)

    def to_json(self) -> dict[str, float]:
        return {"lower": self.lower, "upper": self.upper}


def _as_array(xs: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-d sequence of observations, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("a confidence sequence needs at least one observation")
    if not np.all(np.isfinite(arr)):
        raise ValueError("observations must all be finite")
    if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
        raise ValueError(
            "observations must lie in [0, 1]; normalise with the declared score_scale first"
        )
    return arr


def predmix_eb_lambdas(
    xs: Sequence[float] | NDArray[np.float64],
    alpha: float = 0.05,
    *,
    truncation: float = DEFAULT_TRUNCATION,
    prior_mean: float = DEFAULT_PRIOR_MEAN,
    prior_variance: float = DEFAULT_PRIOR_VARIANCE,
    fake_obs: float = DEFAULT_FAKE_OBS,
) -> NDArray[np.float64]:
    """Predictable-mixture empirical-Bernstein bets.

    ``lambda_t = sqrt( 2 log(1/alpha) / (t log(1+t) sigma_hat^2_{t-1}) )``, truncated.
    Every ``lambda_t`` uses only ``x_1..x_{t-1}``, which is what keeps the resulting
    process a martingale.
    """
    x = _as_array(xs)
    n = x.size
    t = np.arange(1, n + 1, dtype=np.float64)

    mu_hat = np.minimum((fake_obs * prior_mean + np.cumsum(x)) / (t + fake_obs), 1.0)
    sigma2 = (fake_obs * prior_variance + np.cumsum((x - mu_hat) ** 2)) / (t + fake_obs)
    # Shift by one so step t sees only the history.
    sigma2_prev = np.concatenate(([prior_variance], sigma2[: n - 1]))

    with np.errstate(divide="ignore", invalid="ignore"):
        lambdas = np.sqrt(2.0 * math.log(1.0 / alpha) / (t * np.log1p(t) * sigma2_prev))
    lambdas = np.nan_to_num(lambdas, nan=0.0, posinf=truncation, neginf=0.0)
    return np.minimum(lambdas, truncation)


# ---------------------------------------------------------------------------------------
# Hedged capital confidence sequence
# ---------------------------------------------------------------------------------------


def hedged_capital(
    xs: Sequence[float] | NDArray[np.float64],
    m: float,
    alpha: float = 0.05,
    *,
    theta: float = 0.5,
    trunc_scale: float = DEFAULT_TRUNCATION,
) -> NDArray[np.float64]:
    """log of the hedged capital process testing ``H0: mean = m``, after each observation.

    ``max(theta*K_up, (1-theta)*K_down)``, which is dominated by their convex combination
    and therefore a valid e-process by Ville.

    The bets are generated *untruncated* and then clipped to the m-dependent safe range
    ``[-s/(1-m), s/m]`` (and its mirror for the down leg). That single clip is what keeps
    every multiplier at or above ``1 - s > 0``. Pre-truncating the bets at a fixed value
    as well would cap the up leg far below what m allows, weakening it asymmetrically —
    which is exactly the bug the Phase 1.3 differential test caught.
    """
    x = _as_array(xs)
    if not 0.0 < m < 1.0:
        # A null mean at or outside the boundary is refuted by any observation strictly
        # inside it; report infinite evidence rather than dividing by zero.
        return np.full(x.size, np.inf)

    lam_up = np.clip(
        predmix_eb_lambdas(x, alpha * theta, truncation=math.inf),
        -trunc_scale / (1.0 - m),
        trunc_scale / m,
    )
    lam_down = np.clip(
        predmix_eb_lambdas(x, alpha * (1.0 - theta), truncation=math.inf),
        -trunc_scale / m,
        trunc_scale / (1.0 - m),
    )

    log_up = np.cumsum(np.log(1.0 + lam_up * (x - m)))
    log_down = np.cumsum(np.log(1.0 - lam_down * (x - m)))
    return np.maximum(math.log(theta) + log_up, math.log(1.0 - theta) + log_down)


def hedged_cs(
    xs: Sequence[float] | NDArray[np.float64],
    alpha: float = 0.05,
    *,
    breaks: int = DEFAULT_BREAKS,
    theta: float = 0.5,
    trunc_scale: float = DEFAULT_TRUNCATION,
    running_intersection: bool = True,
) -> list[Interval]:
    """Invert the hedged capital martingale over a grid of candidate means.

    ``CS_t = { m : K_t(m) <= 1/alpha }``. By Ville, the true mean's capital process
    exceeds ``1/alpha`` at some time with probability at most ``alpha``, so the true mean
    is excluded at some time with probability at most ``alpha`` — time-uniformly.

    The interval is widened by one grid step on each side, because a grid search only
    knows the boundary to within its own resolution and rounding inward would understate
    the uncertainty.
    """
    x = _as_array(xs)
    n = x.size
    step = 1.0 / breaks
    grid = np.arange(0.0, 1.0 + step, step)
    # Compared in log space, so a candidate whose capital lands exactly on 1/alpha can
    # round either side of the threshold. The slack keeps such a tie *inside* the
    # interval, which is the conservative direction (a wider interval never overstates
    # what we know) and matches the reference's inclusive comparison.
    threshold = math.log(1.0 / alpha) + _LOG_TIE_SLACK

    # log-capital for every (candidate mean, time). breaks x n, which is small enough to
    # hold for the stream lengths this tool deals with.
    capital = np.empty((grid.size, n), dtype=np.float64)
    for i, m in enumerate(grid):
        capital[i] = hedged_capital(x, float(m), alpha, theta=theta, trunc_scale=trunc_scale)

    included = capital <= threshold
    out: list[Interval] = []
    lower_run, upper_run = 0.0, 1.0
    for t in range(n):
        idx = np.flatnonzero(included[:, t])
        if idx.size == 0:
            # No candidate survives: report the whole space rather than an empty or
            # inverted interval, matching the reference's convention.
            lo, hi = 0.0, 1.0
        else:
            lo = max(0.0, float(grid[idx[0]]) - step)
            hi = min(1.0, float(grid[idx[-1]]) + step)
        if running_intersection:
            lower_run = max(lower_run, lo)
            upper_run = min(upper_run, hi)
            lo, hi = lower_run, min(upper_run, 1.0)
            if hi < lo:  # pragma: no cover - only under a coverage failure
                lo = hi = 0.5 * (lo + hi)
        out.append(Interval(lo, hi))
    return out


# ---------------------------------------------------------------------------------------
# Predictable-mixture empirical-Bernstein confidence sequence
# ---------------------------------------------------------------------------------------


def _eb_lower(
    xs: NDArray[np.float64], alpha: float, truncation: float, running_intersection: bool
) -> NDArray[np.float64]:
    """One-sided empirical-Bernstein lower bound, the primitive for the two-sided CS."""
    n = xs.size
    t = np.arange(1, n + 1, dtype=np.float64)
    mu_hat = np.minimum(np.cumsum(xs) / t, 1.0)
    mu_prev = np.concatenate(([0.0], mu_hat[: n - 1]))
    v = (xs - mu_prev) ** 2

    lambdas = predmix_eb_lambdas(xs, alpha, truncation=truncation)
    psi = -np.log1p(-lambdas) - lambdas  # psi_E, up to the factor absorbed into v

    denom = np.cumsum(lambdas)
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = (math.log(1.0 / alpha) + np.cumsum(v * psi)) / denom
        weighted_mean = np.cumsum(lambdas * xs) / denom
    lower = np.nan_to_num(weighted_mean - margin, nan=0.0, neginf=0.0, posinf=1.0)
    lower = np.maximum(lower, 0.0)
    return np.maximum.accumulate(lower) if running_intersection else lower


def empirical_bernstein_cs(
    xs: Sequence[float] | NDArray[np.float64],
    alpha: float = 0.05,
    *,
    truncation: float = DEFAULT_TRUNCATION,
    running_intersection: bool = True,
) -> list[Interval]:
    """Closed-form predictable-mixture empirical-Bernstein confidence sequence.

    The two sides each spend ``alpha/2``, by a union bound.
    """
    x = _as_array(xs)
    lower = _eb_lower(x, alpha / 2.0, truncation, running_intersection)
    upper = 1.0 - _eb_lower(1.0 - x, alpha / 2.0, truncation, running_intersection)
    return [
        Interval(float(min(lo, hi)), float(max(lo, hi)))
        for lo, hi in zip(lower, upper, strict=True)
    ]


# ---------------------------------------------------------------------------------------
# The interface the rest of benchlock uses
# ---------------------------------------------------------------------------------------


def mean_cs(
    xs: Sequence[float] | NDArray[np.float64],
    alpha: float = 0.05,
    *,
    method: str = "hedged",
    breaks: int = DEFAULT_BREAKS,
) -> list[Interval]:
    """Time-uniform confidence sequence for the mean, one interval per observation."""
    if method == "hedged":
        return hedged_cs(xs, alpha, breaks=breaks)
    if method == "empirical-bernstein":
        return empirical_bernstein_cs(xs, alpha)
    raise ValueError(f"unknown CS method {method!r}; use 'hedged' or 'empirical-bernstein'")


def final_interval(
    xs: Sequence[float] | NDArray[np.float64], alpha: float = 0.05, *, method: str = "hedged"
) -> Interval:
    """The interval after the last observation. Still time-uniformly valid."""
    return mean_cs(xs, alpha, method=method)[-1]
