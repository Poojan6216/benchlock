"""Sizing the anchor set: the design law, made arithmetic.

    **Design law.** The anchor process must be provisioned to reach its threshold no later
    than the system process would, for any judge shift large enough to move the system
    stream detectably.

If it is not, a judge shift moves the system stream before the anchor stream can prove it,
and a naive implementation returns ``SYSTEM`` — a confident, wrong rollback recommendation.
Lattice rule 6 enforces this at runtime; ``benchlock plan`` enforces it at design time,
using the arithmetic here.

**How the noise scales with anchor size.** Not as ``1/sqrt(n)``. A judge's run-to-run
variation has two parts: an independent per-item component that does shrink that way, and a
*shared* component that moves every item together — a provider-side change, a different
sampling kernel, a load-shedding fallback. The shared part does not shrink at all. We
decompose the measured floor into both::

    run_mean_sd^2  =  shared_sd^2  +  per_item_sd^2 / n_items

and then a larger anchor set only buys down the second term. That matters: where the shared
component alone already exceeds the target, **no anchor size is sufficient**, and saying so
is far more useful than returning a big number that would not have worked.

**Every rounding goes the conservative way.** Sizes round up, detectable shifts round up,
and the search returns the larger n whenever it is ambiguous. Under-provisioning produces
confidently wrong verdicts; over-provisioning costs judge tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import t as student_t

from benchlock.model.pins import NoiseFloor
from benchlock.stats.eprocess import DEFAULT_BASELINE_ALPHA_FRACTION, split_alpha

#: Extra evidence demanded beyond the bare threshold, as a multiple. The growth model
#: below is a second-order approximation with oracle bets; the real process pays for
#: plug-in bets that must first learn the drift, and for the change-point prior's weight.
#: Calibrated against simulation in `tests/test_power.py`, which re-measures it and fails
#: if the formula and reality part company. Raising it makes the tool ask for a bigger
#: anchor set, which is the safe direction.
DETECTION_SAFETY_FACTOR = 2.6

#: Scaling band, in run-to-run standard deviations. Measured, not assumed: at 10 the bets
#: saturate their truncation and most of the power is thrown away; at 2 ordinary noise
#: clips against the boundary and the signal shape is destroyed. 4 was the best of
#: {2,3,4,5,6,10} across four anchor/replicate/horizon configurations.
DEFAULT_BAND_SDS = 4.0


@dataclass(frozen=True, slots=True)
class NoiseComponents:
    """The measured floor split into what a bigger anchor set can and cannot fix."""

    shared_sd: float  # moves every item together; immune to anchor size
    per_item_sd: float  # independent across items; shrinks as 1/sqrt(n)
    measured_at_n: int

    def run_mean_sd(self, n: int) -> float:
        """Predicted run-mean SD for an anchor set of `n` items."""
        if n < 1:
            raise ValueError(f"anchor size must be at least 1, got {n}")
        return math.sqrt(self.shared_sd**2 + (self.per_item_sd**2) / n)

    @property
    def floor_shift(self) -> float:
        """The run-mean SD no anchor size can go below."""
        return self.shared_sd


def decompose(floor: NoiseFloor) -> NoiseComponents:
    """Split a measured floor into its shared and independent parts.

    A measured ``run_mean_sd`` below what independent items alone predict is sampling
    noise in the estimate, not evidence of negative shared variance; it clamps to zero.
    """
    independent_part = (floor.per_item_sd**2) / floor.n_items
    shared_var = max(0.0, floor.run_mean_sd**2 - independent_part)
    return NoiseComponents(
        shared_sd=math.sqrt(shared_var),
        per_item_sd=floor.per_item_sd,
        measured_at_n=floor.n_items,
    )


def snapshot_half_width(
    run_mean_sd: float,
    replicates: int,
    alpha: float,
    *,
    baseline_alpha_fraction: float = DEFAULT_BASELINE_ALPHA_FRACTION,
) -> float:
    """How far the frozen snapshot may sit from the judge's true mean.

    This is dead zone: a shift smaller than this is inside the null and can never be
    detected, however long you watch.
    """
    if replicates < 2:
        raise ValueError(
            f"a frozen baseline needs at least K=2 replicates, got {replicates}; "
            "set `anchor.noise_replicates` to 2 or more"
        )
    alpha_baseline = alpha * baseline_alpha_fraction
    quantile = float(student_t.ppf(1.0 - alpha_baseline / 2.0, df=replicates - 1))
    return quantile * run_mean_sd / math.sqrt(replicates)


def _growth_rate(drift_z: float, sigma_z: float, lam_max: float) -> float:
    """Expected log-wealth gained per run, in scaled units.

    The bet is the growth-optimal ``drift / (sigma^2 + drift^2)`` *truncated* at
    ``lam_max``, and that truncation is usually what binds. It matters which regime you
    are in: untruncated, wealth grows like ``drift^2 / 2 sigma^2`` — quadratic in the
    drift; truncated, it grows like ``lam_max * drift`` — merely linear. Assuming the
    quadratic rate makes the calculator claim shifts are detectable that are not, which
    is the single most dangerous error this file could contain.
    """
    if drift_z <= 0.0:
        return 0.0
    lam = min(drift_z / (sigma_z * sigma_z + drift_z * drift_z), lam_max)
    centre = 1.0 + lam * drift_z
    return math.log(centre) - (lam * lam * sigma_z * sigma_z) / (2.0 * centre * centre)


def min_detectable_shift(
    anchor_n: int,
    noise_floor: NoiseFloor,
    alpha: float,
    horizon: int,
    *,
    replicates: int | None = None,
    safety: float = DETECTION_SAFETY_FACTOR,
    band_sds: float = DEFAULT_BAND_SDS,
) -> float:
    """The smallest judge shift an anchor set of `anchor_n` items could catch by `horizon`.

    Two things must be cleared: the dead zone left by snapshot measurement error, which no
    amount of watching can overcome, and enough accumulated evidence to cross
    ``1/alpha_monitor`` inside the horizon. Solved numerically because the growth rate
    changes regime when the bet hits its truncation. Rounded up.
    """
    if anchor_n < 1:
        raise ValueError(f"anchor size must be at least 1, got {anchor_n}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 run, got {horizon}")
    K = replicates if replicates is not None else noise_floor.replicates
    components = decompose(noise_floor)
    sigma = components.run_mean_sd(anchor_n)
    dead_zone = snapshot_half_width(sigma, K, alpha)

    _, alpha_monitor = split_alpha(alpha)
    # log(2) for the hedged 50/50 split across the two directions.
    required = safety * (math.log(1.0 / alpha_monitor) + math.log(2.0))

    band = band_sds * sigma
    sigma_z = sigma / (2.0 * band)
    # Testing a null just inside the interval edge, so lam_max is very close to 2c.
    lam_max = 0.5 / (1.0 - 0.5)

    def reaches(shift: float) -> bool:
        drift_z = (shift - dead_zone) / (2.0 * band)
        # Beyond the band the observation clips, so no extra evidence accrues per run.
        drift_z = min(drift_z, 0.5)
        return horizon * _growth_rate(drift_z, sigma_z, lam_max) >= required

    hi = dead_zone + 2.0 * band
    if not reaches(hi):
        # NO shift is provable in this horizon — not even one that saturates the band.
        # Returning a finite magnitude here would be read by every caller as "detectable",
        # because they all compare numerically (`mds < observed`, `mds <= target`). That
        # turned an anchor process which could not possibly cross into an ADEQUATE
        # provisioning verdict, and produced a confident `SYSTEM` rollback recommendation
        # on a pure judge shift — the exact phantom regression this tool exists to prevent.
        # Infinity is the honest answer: no shift is small enough to be caught.
        return math.inf
    lo = dead_zone
    for _ in range(80):  # bisection to well below float noise on a [0,1] score
        mid = 0.5 * (lo + hi)
        if reaches(mid):
            hi = mid
        else:
            lo = mid
    return hi  # the upper end of the bracket: conservative, always the larger shift


def min_anchor_size(
    target_shift: float,
    noise_floor: NoiseFloor,
    alpha: float,
    horizon: int,
    obs_per_run: int,
    *,
    replicates: int | None = None,
    max_n: int = 100_000,
    safety: float = DETECTION_SAFETY_FACTOR,
    band_sds: float = DEFAULT_BAND_SDS,
) -> int:
    """Minimum anchor set size satisfying the design law.

    Conservative: when the numeric search is ambiguous, return the LARGER n.

    Raises `ProvisioningImpossibleError` when no anchor size suffices, which happens when
    the judge's shared noise component alone swamps the target. That is a real answer —
    the alternative is handing back a number that would not have worked.
    """
    if target_shift <= 0.0:
        raise ValueError(f"target_shift must be positive, got {target_shift}")
    if obs_per_run < 1:
        raise ValueError(f"obs_per_run must be at least 1, got {obs_per_run}")

    # 1. Detectability: the anchor must be able to see a shift this small within horizon.
    lo, hi = 1, 1
    while hi <= max_n:
        if (
            min_detectable_shift(
                hi,
                noise_floor,
                alpha,
                horizon,
                replicates=replicates,
                safety=safety,
                band_sds=band_sds,
            )
            <= target_shift
        ):
            break
        lo, hi = hi, hi * 2
    else:
        components = decompose(noise_floor)
        raise ProvisioningImpossibleError(target_shift, components, horizon, alpha)

    while lo < hi:
        mid = (lo + hi) // 2
        if (
            min_detectable_shift(
                mid,
                noise_floor,
                alpha,
                horizon,
                replicates=replicates,
                safety=safety,
                band_sds=band_sds,
            )
            <= target_shift
        ):
            hi = mid
        else:
            lo = mid + 1
    detectability_n = lo

    # 2. The design law proper: the anchor must be no noisier than the system stream, or a
    #    judge shift reaches the system's threshold first and gets blamed on the system.
    #    The system stream sees the same judge over `obs_per_run` items per run, so that is
    #    the noise the anchor has to match.
    components = decompose(noise_floor)
    system_sigma = components.run_mean_sd(obs_per_run)
    race_n = detectability_n
    while race_n < max_n and components.run_mean_sd(race_n) > system_sigma:
        race_n += 1

    return max(detectability_n, race_n)


def min_feasible_horizon(alpha: float, *, safety: float = DETECTION_SAFETY_FACTOR) -> int:
    """The fewest runs in which ANY shift could be proven, at any anchor size.

    Each run's log-wealth gain is capped by the bet truncation at ``log(1 + c)`` — a
    saturating observation cannot contribute more than that. Crossing needs
    ``safety * (log(1/alpha_m) + log 2)`` nats. The ratio is a floor on the horizon that no
    amount of anchor provisioning can lower, and it is worth naming because a user who
    asks for detection inside ten runs is asking for something arithmetic forbids.
    """
    _, alpha_monitor = split_alpha(alpha)
    required = safety * (math.log(1.0 / alpha_monitor) + math.log(2.0))
    per_run_cap = math.log(1.5)  # c = 0.5: the multiplier can never exceed 1 + c
    return math.ceil(required / per_run_cap)


class ProvisioningImpossibleError(Exception):
    """No anchor size can reach the target. The message says which constraint binds."""

    def __init__(
        self, target_shift: float, components: NoiseComponents, horizon: int, alpha: float
    ) -> None:
        self.target_shift = target_shift
        self.components = components
        floor = min_feasible_horizon(alpha)
        if horizon < floor:
            self.message = (
                f"no anchor size can detect a judge shift of {target_shift:g} within "
                f"{horizon} runs at alpha={alpha:g}, because nothing at all is provable in "
                f"fewer than {floor} runs: each run's evidence is capped by the bet "
                "truncation, so a crossing needs at least that many runs whatever the judge "
                "did"
            )
            self.hint = (
                f"lengthen --horizon to at least {floor}, or raise alpha. More anchor items "
                "cannot help here"
            )
        else:
            self.message = (
                f"no anchor size can detect a judge shift of {target_shift:g} within "
                f"{horizon} runs at alpha={alpha:g}. The judge's shared run-to-run noise is "
                f"{components.shared_sd:.4f}, which moves every anchor item together and so "
                "does not shrink as the anchor set grows"
            )
            self.hint = (
                "raise --target-shift, lengthen --horizon, increase "
                "`anchor.noise_replicates` so the snapshot is measured more precisely, or "
                "reduce the judge's own variability (a more decisive rubric, a pinned "
                "snapshot)"
            )
        super().__init__(f"{self.message}\n  fix: {self.hint}")


@dataclass(frozen=True, slots=True)
class Plan:
    """What `benchlock plan` reports."""

    anchor_n: int
    cadence: int
    horizon: int
    target_shift: float
    achieved_min_detectable_shift: float
    dead_zone: float
    shared_sd: float
    per_item_sd: float
    replicates: int
    obs_per_run: int
    alpha: float
    #: Fractional improvement in detectable shift from quadrupling the anchor set.
    marginal_gain_at_4x: float

    @property
    def shared_noise_dominates(self) -> bool:
        """True when buying more anchor items would barely move the detectable shift.

        Defined by diminishing returns rather than by comparing the shared component to
        the target, because what a user needs to know is whether spending more judge
        tokens will help. Below 10% improvement for a 4x larger anchor set, the answer is
        no: the judge's shared run-to-run movement is the binding constraint, and the fix
        is a steadier judge, more replicates, or a longer horizon.
        """
        return self.marginal_gain_at_4x < 0.10

    def to_json(self) -> dict[str, float | int | bool]:
        return {
            "anchor_n": self.anchor_n,
            "cadence": self.cadence,
            "horizon": self.horizon,
            "target_shift": self.target_shift,
            "achieved_min_detectable_shift": self.achieved_min_detectable_shift,
            "dead_zone": self.dead_zone,
            "shared_sd": self.shared_sd,
            "per_item_sd": self.per_item_sd,
            "replicates": self.replicates,
            "obs_per_run": self.obs_per_run,
            "alpha": self.alpha,
            "marginal_gain_at_4x": self.marginal_gain_at_4x,
            "shared_noise_dominates": self.shared_noise_dominates,
        }


def make_plan(
    target_shift: float,
    noise_floor: NoiseFloor,
    alpha: float,
    horizon: int,
    obs_per_run: int,
    *,
    cadence: int = 1,
) -> Plan:
    """Size an anchor set and report what it will and will not be able to see."""
    n = min_anchor_size(target_shift, noise_floor, alpha, horizon, obs_per_run)
    components = decompose(noise_floor)
    sigma = components.run_mean_sd(n)
    return Plan(
        anchor_n=n,
        cadence=cadence,
        horizon=horizon,
        target_shift=target_shift,
        achieved_min_detectable_shift=min_detectable_shift(n, noise_floor, alpha, horizon),
        dead_zone=snapshot_half_width(sigma, noise_floor.replicates, alpha),
        shared_sd=components.shared_sd,
        per_item_sd=components.per_item_sd,
        replicates=noise_floor.replicates,
        obs_per_run=obs_per_run,
        alpha=alpha,
        marginal_gain_at_4x=_marginal_gain(n, noise_floor, alpha, horizon),
    )


def _marginal_gain(n: int, noise_floor: NoiseFloor, alpha: float, horizon: int) -> float:
    """Fractional improvement in detectable shift from quadrupling the anchor set."""
    here = min_detectable_shift(n, noise_floor, alpha, horizon)
    bigger = min_detectable_shift(n * 4, noise_floor, alpha, horizon)
    if here <= 0.0:  # pragma: no cover - the dead zone keeps this positive
        return 0.0
    return (here - bigger) / here
