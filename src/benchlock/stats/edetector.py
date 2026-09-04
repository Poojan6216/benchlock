"""Change detection built from e-processes, with bounded memory.

A drift can begin at any run, so a single e-process anchored at time zero is the wrong
statistic: by the time a late change arrives, the process has already spent its evidence
on a period where nothing was happening. The fix is to hypothesise a change point at every
run and combine::

    E_t = sum_j  w_j * E_j(t)

where ``E_j`` is an e-process started at run ``j`` and the weights ``w_j`` sum to one. The
expectation of the sum is at most one under the null, so the combination is itself an
e-value and Ville's inequality applies to it directly.

**Why a weighted sum rather than the unweighted one.** The e-detectors of Shin, Ramdas &
Rinaldo sum without weights and earn an average-run-length guarantee: ``E[tau] >= 1/alpha``.
That is the classical change-detection promise, and it is weaker than what Hard Rule 3
demands. Weighting so the prior over change points sums to one buys the stronger
time-uniform statement — ``P(ever alarming on a drift-free stream) <= alpha`` — at the cost
of some power against very late change points. For a CI gate that is inspected on every
commit, bounding the probability of *ever* crying wolf is the promise worth having.

**Bounded memory, conservatively (Hard Rule 4).** Retaining every candidate costs memory
linear in run count. We keep at most ``max_candidates``, dropping those contributing least.
Because every term is non-negative and survivors keep their original weights, the retained
sum is a sub-sum of the full one::

    E_bounded(t) = sum_{j retained} w_j E_j(t)  <=  sum_{all j} w_j E_j(t) = E_full(t)

so pruning can only ever *delay* an alarm, never create one. The property test in
``tests/test_edetector.py`` verifies this over thousands of streams, and it is a
build-blocking test — but note that the inequality holds by construction, not by luck.
Re-weighting survivors upward after a prune would break it, which is precisely why we
never do that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from benchlock.stats.betting import BettingStrategy, Side
from benchlock.stats.eprocess import BaselineNull, CompositeNullEProcess

#: Default cap on retained change-point hypotheses.
DEFAULT_MAX_CANDIDATES = 256


def changepoint_log_weight(j: int) -> float:
    """log of the prior mass on 'the change began at run j', for 0-based j.

    ``w_j = 1 / ((j+1)(j+2))`` telescopes to exactly 1 over all j >= 0, so the weights are
    a genuine probability distribution and the weighted sum is a valid e-value with no
    slack left unaccounted for.
    """
    if j < 0:
        raise ValueError(f"change point index must be >= 0, got {j}")
    return -math.log(j + 1.0) - math.log(j + 2.0)


@dataclass(slots=True)
class Candidate:
    """One change-point hypothesis and the e-process testing it."""

    start: int  # run index at which the hypothesised change begins
    log_weight: float
    process: CompositeNullEProcess

    @property
    def log_contribution(self) -> float:
        """log(w_j * E_j) — what this hypothesis adds to the combined statistic."""
        return self.log_weight + self.process.log_e


@dataclass(slots=True)
class DetectorState:
    """Everything a report needs about the detector, with no statistics recomputed."""

    log_e: float
    t: int
    crossed_at: int | None
    best_changepoint: int | None
    n_candidates: int
    n_pruned: int
    direction: Side | None

    @property
    def e_value(self) -> float:
        try:
            return math.exp(self.log_e)
        except OverflowError:  # pragma: no cover
            return math.inf


def _log_sum_exp(values: Sequence[float]) -> float:
    """Stable log of a sum of exponentials. Returns -inf for an empty sequence."""
    if not values:
        return -math.inf
    hi = max(values)
    if hi == -math.inf:
        return -math.inf
    return hi + math.log(sum(math.exp(v - hi) for v in values))


class EDetector:
    """Sequential change detection over an unknown change point, with bounded memory."""

    __slots__ = (
        "_alarm_time",
        "_c",
        "_candidates",
        "_n_pruned",
        "_strategy",
        "_t",
        "alpha",
        "max_candidates",
        "null",
    )

    def __init__(
        self,
        null: BaselineNull,
        alpha: float,
        *,
        max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
        strategy: BettingStrategy | None = None,
        c: float = 0.5,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if max_candidates is not None and max_candidates < 1:
            raise ValueError(f"max_candidates must be at least 1, got {max_candidates}")
        self.null = null
        self.alpha = alpha
        self.max_candidates = max_candidates
        self._strategy = strategy
        self._c = c
        self._candidates: list[Candidate] = []
        self._t = 0
        self._alarm_time: int | None = None
        self._n_pruned = 0

    # ---- the loop ----------------------------------------------------------------------

    def update(self, x: float) -> float:
        """Fold in one run's observation and return the combined log e-value."""
        # A change could begin at this run, so open a hypothesis for it first.
        self._candidates.append(
            Candidate(
                start=self._t,
                log_weight=changepoint_log_weight(self._t),
                process=CompositeNullEProcess(
                    self.null,
                    self.alpha,
                    strategy=self._strategy,
                    c=self._c,
                ),
            )
        )
        for candidate in self._candidates:
            candidate.process.update(x)
        self._t += 1
        self._prune()

        if self._alarm_time is None and self.crossed():
            self._alarm_time = self._t - 1
        return self.log_e

    def update_many(self, xs: Sequence[float]) -> float:
        for x in xs:
            self.update(x)
        return self.log_e

    def _prune(self) -> None:
        """Drop the least-contributing hypotheses. Survivors keep their weights untouched.

        Re-weighting survivors upward would redistribute the pruned mass and could push
        the statistic above what full memory would have reported — the exact failure Hard
        Rule 4 forbids. So we simply drop terms from a sum of non-negative numbers.
        """
        if self.max_candidates is None or len(self._candidates) <= self.max_candidates:
            return
        self._candidates.sort(key=lambda cand: cand.log_contribution, reverse=True)
        self._n_pruned += len(self._candidates) - self.max_candidates
        del self._candidates[self.max_candidates :]

    # ---- the statistic -----------------------------------------------------------------

    @property
    def log_e(self) -> float:
        return _log_sum_exp([cand.log_contribution for cand in self._candidates])

    @property
    def e_value(self) -> float:
        try:
            return math.exp(self.log_e)
        except OverflowError:  # pragma: no cover
            return math.inf

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha

    def crossed(self) -> bool:
        return self.log_e >= math.log(self.threshold)

    @property
    def alarm_time(self) -> int | None:
        """0-based run index at which the detector first crossed, or None."""
        return self._alarm_time

    @property
    def best_changepoint(self) -> int | None:
        """Where the change most likely began. **Descriptive only** — no verdict reads it.

        Ranked by each hypothesis's own evidence, *not* by its weighted contribution. The
        weights are a prior chosen to make the sum a valid e-value, and they fall off
        steeply with j; ranking by weighted contribution would therefore report the
        earliest candidate almost regardless of the data, which is a statement about the
        prior rather than about the stream.
        """
        if not self._candidates:
            return None
        return max(self._candidates, key=lambda cand: cand.process.log_e).start

    @property
    def direction(self) -> Side | None:
        if not self._candidates:
            return None
        best = max(self._candidates, key=lambda cand: cand.process.log_e)
        return best.process.direction

    @property
    def n_candidates(self) -> int:
        return len(self._candidates)

    @property
    def n_pruned(self) -> int:
        return self._n_pruned

    def state(self) -> DetectorState:
        return DetectorState(
            log_e=self.log_e,
            t=self._t,
            crossed_at=self._alarm_time,
            best_changepoint=self.best_changepoint,
            n_candidates=self.n_candidates,
            n_pruned=self._n_pruned,
            direction=self.direction,
        )


@dataclass(frozen=True, slots=True)
class DetectorTrace:
    """The whole trajectory, for plots and for the report's e-value traces."""

    log_e: tuple[float, ...] = field(default_factory=tuple)
    crossed_at: int | None = None

    @property
    def e_values(self) -> tuple[float, ...]:
        return tuple(min(math.exp(v), math.inf) if v < 700 else math.inf for v in self.log_e)


def run_detector(
    xs: Sequence[float],
    null: BaselineNull,
    alpha: float,
    *,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    strategy: BettingStrategy | None = None,
) -> tuple[EDetector, DetectorTrace]:
    """Run a detector over a whole stream, keeping the trace."""
    detector = EDetector(null, alpha, max_candidates=max_candidates, strategy=strategy)
    trace: list[float] = []
    for x in xs:
        detector.update(x)
        trace.append(detector.log_e)
    return detector, DetectorTrace(log_e=tuple(trace), crossed_at=detector.alarm_time)
