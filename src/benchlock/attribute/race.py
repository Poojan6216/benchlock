"""The anchor race: could the anchor set have seen a judge shift this big?

    **Design law.** The anchor process must be provisioned to reach its threshold no later
    than the system process would, for any judge shift large enough to move the system
    stream detectably.

When it is not, the failure is silent and expensive. A judge shift moves *both* streams,
but if the anchor set is too small it moves the system stream past its threshold first,
the anchor stream stays quiet, and the obvious reading — "the system moved and the judge
didn't" — is exactly wrong. The team rolls back a healthy release.

`benchlock plan` enforces the law at design time. This module enforces it at run time, by
asking one question of every would-be ``SYSTEM`` verdict: *given the anchor set we
actually have and the runs we have actually seen, could a judge shift the size of the
observed system move have been detected by now?* If the answer is no, the honest verdict
is ``INDETERMINATE``, not ``SYSTEM``.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchlock.model.pins import NoiseFloor
from benchlock.model.verdict import Provisioning
from benchlock.stats.power import decompose, min_detectable_shift


@dataclass(frozen=True, slots=True)
class RaceCheck:
    """The provisioning verdict for one decision."""

    min_detectable_judge_shift: float
    observed_system_shift: float
    provisioning: Provisioning
    anchor_n: int
    runs_elapsed: int
    shared_noise_sd: float

    @property
    def adequate(self) -> bool:
        return self.provisioning is Provisioning.ADEQUATE

    def explain(self) -> str:
        if self.adequate:
            return (
                f"anchor provisioning: ADEQUATE\n"
                f"  minimum detectable judge shift at this anchor size = "
                f"{self.min_detectable_judge_shift:.3f} < observed "
                f"{abs(self.observed_system_shift):.3f}"
            )
        return (
            f"anchor provisioning: UNDER_PROVISIONED\n"
            f"  at n={self.anchor_n} the minimum judge shift this anchor process could "
            f"have detected by run {self.runs_elapsed} is "
            f"{self.min_detectable_judge_shift:.3f}. The observed system move is "
            f"{abs(self.observed_system_shift):.3f}. A judge shift of that size would be "
            f"INVISIBLE here."
        )


def check_race(
    observed_system_shift: float,
    noise_floor: NoiseFloor,
    anchor_n: int,
    runs_elapsed: int,
    alpha: float,
    *,
    target_shift: float | None = None,
) -> RaceCheck:
    """Decide whether the anchor process had the power to rule the judge out.

    ``runs_elapsed`` is the horizon the anchor process has actually had — not the horizon
    it was planned for. A well-provisioned anchor set is still under-provisioned on run 3.
    """
    if anchor_n < 1:
        raise ValueError(f"anchor size must be at least 1, got {anchor_n}")
    horizon = max(1, runs_elapsed)
    band_target = target_shift if target_shift is not None else abs(observed_system_shift)
    mds = min_detectable_shift(
        anchor_n,
        noise_floor,
        alpha,
        horizon,
        band_sds=_band_sds_for(band_target, noise_floor, anchor_n),
    )
    adequate = mds < abs(observed_system_shift)
    return RaceCheck(
        min_detectable_judge_shift=mds,
        observed_system_shift=observed_system_shift,
        provisioning=Provisioning.ADEQUATE if adequate else Provisioning.UNDER_PROVISIONED,
        anchor_n=anchor_n,
        runs_elapsed=runs_elapsed,
        shared_noise_sd=decompose(noise_floor).shared_sd,
    )


def _band_sds_for(target_shift: float, noise_floor: NoiseFloor, anchor_n: int) -> float:
    """The band the monitoring stream is actually scaled with, expressed in SDs.

    Kept consistent with `MonitorScale.for_target`, so the power calculation describes the
    detector that is really running rather than an idealised one.
    """
    sigma = decompose(noise_floor).run_mean_sd(anchor_n)
    if sigma <= 0.0:  # pragma: no cover - MIN_SD keeps the floor positive
        return 4.0
    return max(4.0, target_shift / sigma)
