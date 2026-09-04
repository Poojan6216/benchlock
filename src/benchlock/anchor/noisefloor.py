"""Measuring how much a judge disagrees with itself.

Hosted judges are nondeterministic even at ``temperature=0``: batching, kernel selection,
speculative decoding and load-shedding fallbacks all move logits, and rubric prompts that
sit near a decision boundary amplify the result into a different score. So "the anchor
score moved" is not evidence of anything until you know how much it moves when *nothing*
has changed.

That is what the noise floor is. Score the frozen anchor set K times at baseline under an
identical configuration, and measure two things:

* ``per_item_sd`` — how much a single item's score wobbles between identical calls.
* ``run_mean_sd`` — how much the *run mean* wobbles, which is the quantity the detector
  actually monitors. Measured directly rather than derived as ``per_item_sd/sqrt(n)``,
  because a judge that drifts globally between calls moves every item together, and the
  derived figure would understate the real variability by assuming independence.

Every downstream test is against this floor rather than against zero, which is the
difference between a detector that fires constantly and one that means something. Phase
1.5's verify demonstrates exactly that.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from benchlock.model.pins import NoiseFloor

#: Smallest standard deviation we will report. A judge that produced K identical scorings
#: has not proven it has zero noise — it has proven K was too small to see any. Using a
#: literal zero would make the monitored band collapse and every later run read as drift.
MIN_SD = 1e-6


class NoiseFloorError(Exception):
    """Raised when replicates cannot support a noise-floor estimate at all."""

    def __init__(self, message: str, hint: str = "") -> None:
        self.message = message
        self.hint = hint
        super().__init__(f"{message}" + (f"\n  fix: {hint}" if hint else ""))


@dataclass(frozen=True, slots=True)
class NoiseFloorReport:
    """The floor itself, plus the diagnostics a human needs to trust it.

    ``NoiseFloor`` is hashed into the anchor pin, so it stays a small fixed record; the
    diagnostics that inform a person rather than the arithmetic live out here.
    """

    floor: NoiseFloor
    #: Mean |difference| between two independent scorings of the same item.
    mean_abs_pairwise_diff: float
    #: Fraction of (item, replicate-pair) comparisons that agreed exactly.
    exact_agreement_rate: float
    #: Largest single-item disagreement seen between any two replicates.
    max_item_disagreement: float
    #: Observed range of the K run means.
    run_mean_range: float
    #: True when the measurement hit MIN_SD, i.e. K replicates showed no variation at all.
    degenerate: bool
    #: Items whose scores varied at all across replicates.
    unstable_items: int

    def warnings(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.degenerate:
            out.append(
                f"all {self.floor.replicates} replicates produced identical scores. Either "
                "the judge is deterministic or K is too small to see its noise. Benchlock "
                f"is using a floor of {MIN_SD:g}, so any later movement will read as drift. "
                "Raise `anchor.noise_replicates` if that is not what you intend"
            )
        if self.floor.replicates < 3:
            out.append(
                f"K={self.floor.replicates} replicates bound the floor very loosely "
                "(a Student-t interval on 1 degree of freedom is enormous). K=5 is the default"
            )
        return tuple(out)

    def to_json(self) -> dict[str, float | int | bool]:
        return {
            "per_item_sd": self.floor.per_item_sd,
            "run_mean_sd": self.floor.run_mean_sd,
            "replicates": self.floor.replicates,
            "n_items": self.floor.n_items,
            "mean_abs_pairwise_diff": self.mean_abs_pairwise_diff,
            "exact_agreement_rate": self.exact_agreement_rate,
            "max_item_disagreement": self.max_item_disagreement,
            "run_mean_range": self.run_mean_range,
            "degenerate": self.degenerate,
            "unstable_items": self.unstable_items,
        }


def estimate_noise_floor(replicates: Sequence[Mapping[str, float]]) -> NoiseFloorReport:
    """Estimate the within-judge noise floor from K identical scorings of the same items.

    Every replicate must cover exactly the same item set; a differing set means the
    replicates are not measuring the same thing, and that is an error rather than
    something to average over (Hard Rule 10).
    """
    K = len(replicates)
    if K < 2:
        raise NoiseFloorError(
            f"a noise floor needs at least 2 replicates, got {K}",
            "set `anchor.noise_replicates` to 5 (the default) and re-run `benchlock baseline`",
        )

    item_sets = [frozenset(r) for r in replicates]
    if len(set(item_sets)) != 1:
        first = item_sets[0]
        differing = next(i for i, s in enumerate(item_sets) if s != first)
        missing = sorted(first - item_sets[differing])[:3]
        extra = sorted(item_sets[differing] - first)[:3]
        raise NoiseFloorError(
            f"replicate {differing} scored a different item set than replicate 0 "
            f"(missing {missing or 'nothing'}, unexpected {extra or 'nothing'})",
            "every replicate must score exactly the same frozen anchor items; otherwise "
            "the variation you measure is the item set changing, not the judge",
        )

    items = sorted(item_sets[0])
    if not items:
        raise NoiseFloorError(
            "the replicates contain no items",
            "freeze a non-empty anchor set before measuring the noise floor",
        )

    # --- per-item variability, pooled across items ---
    per_item_variances: list[float] = []
    abs_diffs: list[float] = []
    agreements = 0
    comparisons = 0
    max_disagreement = 0.0
    unstable = 0
    for item in items:
        scores = [float(r[item]) for r in replicates]
        per_item_variances.append(statistics.variance(scores))
        varied = False
        for a in range(K):
            for b in range(a + 1, K):
                diff = abs(scores[a] - scores[b])
                abs_diffs.append(diff)
                comparisons += 1
                if diff == 0.0:
                    agreements += 1
                else:
                    varied = True
                max_disagreement = max(max_disagreement, diff)
        unstable += int(varied)

    per_item_sd = math.sqrt(sum(per_item_variances) / len(per_item_variances))

    # --- run-mean variability, measured directly so correlated drift is captured ---
    run_means = [sum(float(r[i]) for i in items) / len(items) for r in replicates]
    run_mean_sd = statistics.stdev(run_means)

    degenerate = run_mean_sd <= 0.0
    floor = NoiseFloor(
        per_item_sd=max(per_item_sd, MIN_SD),
        run_mean_sd=max(run_mean_sd, MIN_SD),
        replicates=K,
        n_items=len(items),
    )
    return NoiseFloorReport(
        floor=floor,
        mean_abs_pairwise_diff=sum(abs_diffs) / len(abs_diffs),
        exact_agreement_rate=agreements / comparisons,
        max_item_disagreement=max_disagreement,
        run_mean_range=max(run_means) - min(run_means),
        degenerate=degenerate,
        unstable_items=unstable,
    )


def independence_ratio(floor: NoiseFloor) -> float:
    """``run_mean_sd`` divided by what independent items would predict.

    A ratio near 1 means item noise is roughly independent, so a bigger anchor set buys
    the usual ``1/sqrt(n)`` improvement. A ratio well above 1 means the judge moves items
    together between calls, and enlarging the anchor set will help far less than the
    provisioning formula's independent-items assumption suggests. `benchlock plan` reports
    this so the number is never taken on faith.
    """
    predicted = floor.per_item_sd / math.sqrt(floor.n_items)
    if predicted <= 0.0:  # pragma: no cover - MIN_SD keeps this positive
        return 1.0
    return floor.run_mean_sd / predicted
