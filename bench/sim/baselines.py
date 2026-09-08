"""The methods Benchlock is measured against — implemented properly, not as strawmen.

A benchmark whose baselines are deliberately weak proves nothing. These are what teams
actually run, implemented the way a competent person would implement them:

* **B0 fixed threshold** — "alert if the score drops more than five points". The true
  industry default, and it is not stupid: it is fast, it has no assumptions, and on a
  large enough effect it works.
* **B1 peeking t-test** — a two-sample t-test at alpha=0.05, re-run at every run. The
  sophisticated-looking one. Under optional stopping it has no type-I error control at
  all, which is exactly what Phase 5.5's first headline number measures.
* **B2 Bonferroni-corrected peeking** — B1 divided by the number of looks so far. The
  naive fix, and it is genuinely a fix for the false-alarm rate; the cost shows up in the
  delay column.
* **B3 CUSUM** — Page's test, the classical sequential change detector.
* **B4 ADWIN / DDM** — adaptive windowing and drift detection from `river`.
* **B5 Benchlock without an anchor stream** — the ablation. Identical statistics, no
  control group. The gap between B5 and B6 is what attribution actually buys; if there is
  no gap, the central claim of this project is unsupported (Decision Gate 3).
* **B6 Benchlock, full.**

**B0 through B5 cannot attribute.** They monitor one stream, so the only conclusion
available to them is "something moved". That is a structural fact about single-stream
methods, not a criticism of the people who use them, and the results report it as such.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import stats

from benchlock.attribute.engine import decide
from benchlock.config import AttributionConfig
from benchlock.model.streams import RunRecord
from benchlock.model.verdict import AttributionRefusedError, Verdict

#: What a single-stream method concludes when it fires. It cannot say *what* moved.
REGRESSION = "regression"
STABLE = "stable"


@dataclass(frozen=True, slots=True)
class MethodResult:
    """One method's answer on one stream."""

    name: str
    alarm_time: int | None  # 0-based monitored-run index of the first alarm
    verdict: str

    @property
    def alarmed(self) -> bool:
        return self.alarm_time is not None


@dataclass(frozen=True, slots=True)
class StreamView:
    """What a method is given: run means, plus the anchor stream if it can use one."""

    system_means: tuple[float, ...]
    anchor_means: tuple[float, ...]
    baseline_runs: int
    obs_per_run: int
    #: Per-item scores, for the methods that test at item level rather than run level.
    system_items: tuple[tuple[float, ...], ...] = ()
    system_runs: tuple[RunRecord, ...] = ()
    anchor_runs: tuple[RunRecord, ...] = ()

    @property
    def baseline(self) -> tuple[float, ...]:
        return self.system_means[: self.baseline_runs]

    @property
    def monitored(self) -> tuple[float, ...]:
        return self.system_means[self.baseline_runs :]


class Method(Protocol):
    name: str

    def run(self, view: StreamView) -> MethodResult: ...


# ---------------------------------------------------------------------------------------
# B0 — fixed threshold
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FixedThreshold:
    """ "Alert if it drops more than five points." Five points of a 100-point scale."""

    drop: float = 0.05
    name: str = "B0-fixed-threshold"

    def run(self, view: StreamView) -> MethodResult:
        baseline = float(np.mean(view.baseline))
        for t, mean in enumerate(view.monitored):
            if baseline - mean > self.drop:
                return MethodResult(self.name, t, REGRESSION)
        return MethodResult(self.name, None, STABLE)


# ---------------------------------------------------------------------------------------
# B1 / B2 — the peeking t-test, uncorrected and Bonferroni-corrected
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PeekingTTest:
    """A two-sample t-test re-run at every accumulating observation.

    This is the one that looks principled and is not. Item-level scores from the baseline
    period are compared against everything seen since, at a fixed alpha, after every run.
    """

    alpha: float = 0.05
    bonferroni: bool = False
    name: str = "B1-peeking-t-test"

    def run(self, view: StreamView) -> MethodResult:
        if not view.system_items:
            return MethodResult(self.name, None, STABLE)
        baseline_items = np.concatenate(
            [np.asarray(run) for run in view.system_items[: view.baseline_runs]]
        )
        seen: list[np.ndarray] = []
        for t, run in enumerate(view.system_items[view.baseline_runs :]):
            seen.append(np.asarray(run))
            current = np.concatenate(seen)
            if current.size < 2 or baseline_items.size < 2:
                continue
            if float(np.std(current)) == 0.0 and float(np.std(baseline_items)) == 0.0:
                continue
            _, p_value = stats.ttest_ind(baseline_items, current, equal_var=False)
            alpha = self.alpha / (t + 1) if self.bonferroni else self.alpha
            if float(p_value) < alpha:
                return MethodResult(self.name, t, REGRESSION)
        return MethodResult(self.name, None, STABLE)


def bonferroni_t_test() -> PeekingTTest:
    return PeekingTTest(bonferroni=True, name="B2-bonferroni-t-test")


# ---------------------------------------------------------------------------------------
# B3 — CUSUM (Page's test)
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cusum:
    """Page's cumulative sum, two-sided, with the reference value at half the target shift."""

    target_shift: float = 0.05
    threshold_sds: float = 5.0
    name: str = "B3-cusum"

    def run(self, view: StreamView) -> MethodResult:
        baseline = np.asarray(view.baseline)
        centre = float(np.mean(baseline))
        sd = float(np.std(baseline, ddof=1)) if baseline.size > 1 else 1e-6
        sd = max(sd, 1e-9)
        k = self.target_shift / 2.0  # the classical reference value
        h = self.threshold_sds * sd
        high = low = 0.0
        for t, mean in enumerate(view.monitored):
            deviation = mean - centre
            high = max(0.0, high + deviation - k)
            low = max(0.0, low - deviation - k)
            if high > h or low > h:
                return MethodResult(self.name, t, REGRESSION)
        return MethodResult(self.name, None, STABLE)


# ---------------------------------------------------------------------------------------
# B4 — ADWIN and DDM
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Adwin:
    """ADWIN over the item-level score stream, like every other baseline that can use it.

    It was previously fed one observation per run — the run mean — while the peeking
    t-test and DDM both received individual item scores. ADWIN's cut bound carries an
    absolute term calibrated for values spanning [0,1]; on a sequence of run means, whose
    spread is smaller than the per-item spread by `sqrt(n)`, that term sits above the
    shift being tested and the algorithm is structurally blind to its own input. Reporting
    a named third-party method as detecting nothing, when the reason is the stream we
    chose to hand it, is not a fair comparison. It is given the same stream as the others
    here; it still detects the grid's shifts rarely, and that is now its own result rather
    than an artefact of ours.
    """

    delta: float = 0.05
    name: str = "B4-adwin"

    def run(self, view: StreamView) -> MethodResult:
        from river.drift import ADWIN

        if not view.system_items:
            return MethodResult(self.name, None, STABLE)
        detector = ADWIN(delta=self.delta)
        for run in view.system_items[: view.baseline_runs]:
            for score in run:
                detector.update(float(score))
        for t, run in enumerate(view.system_items[view.baseline_runs :]):
            for score in run:
                detector.update(float(score))
            if detector.drift_detected:
                return MethodResult(self.name, t, REGRESSION)
        return MethodResult(self.name, None, STABLE)


@dataclass(frozen=True, slots=True)
class Ddm:
    """DDM on a binarised stream: an item "fails" if it scores below the baseline mean."""

    name: str = "B4-ddm"

    def run(self, view: StreamView) -> MethodResult:
        from river.drift.binary import DDM

        if not view.system_items:
            return MethodResult(self.name, None, STABLE)
        centre = float(np.mean(view.baseline))
        detector = DDM()
        for run in view.system_items[: view.baseline_runs]:
            for score in run:
                detector.update(int(score < centre))
        for t, run in enumerate(view.system_items[view.baseline_runs :]):
            for score in run:
                detector.update(int(score < centre))
            if detector.drift_detected:
                return MethodResult(self.name, t, REGRESSION)
        return MethodResult(self.name, None, STABLE)


# ---------------------------------------------------------------------------------------
# B5 / B6 — Benchlock without and with the anchor stream
# ---------------------------------------------------------------------------------------


DEFAULT_CONFIG = AttributionConfig()


@dataclass(frozen=True, slots=True)
class BenchlockNoAnchor:
    """The ablation: identical statistics, no control group.

    It can detect that the system stream moved. It cannot say what moved it, so like every
    other single-stream method its only available conclusion is "regression". The gap
    between this and B6 is precisely what the anchor set buys.
    """

    config: AttributionConfig = DEFAULT_CONFIG
    name: str = "B5-benchlock-no-anchor"

    def run(self, view: StreamView) -> MethodResult:
        if not view.system_runs:
            return MethodResult(self.name, None, STABLE)
        try:
            attribution = decide(list(view.system_runs), [], self.config)
        except AttributionRefusedError:
            return MethodResult(self.name, None, STABLE)
        crossed = attribution.evidence.crossed_at_system
        if attribution.evidence.system_crossed:
            return MethodResult(self.name, crossed, REGRESSION)
        return MethodResult(self.name, None, STABLE)


@dataclass(frozen=True, slots=True)
class Benchlock:
    config: AttributionConfig = DEFAULT_CONFIG
    name: str = "B6-benchlock"

    def run(self, view: StreamView) -> MethodResult:
        if not view.system_runs:
            return MethodResult(self.name, None, STABLE)
        try:
            attribution = decide(list(view.system_runs), list(view.anchor_runs), self.config)
        except AttributionRefusedError:
            return MethodResult(self.name, None, "refused")
        alarm = _first_alarm(
            attribution.evidence.crossed_at_system, attribution.evidence.crossed_at_anchor
        )
        if attribution.verdict is Verdict.STABLE:
            alarm = None
        return MethodResult(self.name, alarm, attribution.verdict.value)


def _first_alarm(system_at: int | None, anchor_at: int | None) -> int | None:
    candidates = [c for c in (system_at, anchor_at) if c is not None]
    return min(candidates) if candidates else None


def all_methods(config: AttributionConfig) -> list[Method]:
    """Every method, in the order they appear in the results tables."""
    return [
        FixedThreshold(),
        PeekingTTest(),
        bonferroni_t_test(),
        Cusum(target_shift=config.target_shift),
        Adwin(),
        Ddm(),
        BenchlockNoAnchor(config=config),
        Benchlock(config=config),
    ]


#: Methods that monitor a single stream and therefore cannot attribute. Reported as a
#: structural fact in the results, not as a failing.
SINGLE_STREAM = {
    "B0-fixed-threshold",
    "B1-peeking-t-test",
    "B2-bonferroni-t-test",
    "B3-cusum",
    "B4-adwin",
    "B4-ddm",
    "B5-benchlock-no-anchor",
}


def arl0(alarm_times: Sequence[int | None], horizon: int) -> float:
    """Average run length to a false alarm, on drift-free streams.

    Streams that never alarm are censored at the horizon, so this is a lower bound on the
    true ARL0 — which is the honest direction, and is stated wherever the number appears.
    """
    if not alarm_times:
        return math.nan
    return float(np.mean([t if t is not None else horizon for t in alarm_times]))
