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

import numpy as np
from numpy.typing import NDArray

from benchlock.stats.betting import DEFAULT_TRUNCATION, AgrapaBet, BettingStrategy, Side
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


class ReferenceEDetector:
    """The scalar reference implementation: one `CompositeNullEProcess` object per candidate.

    Kept because it is the readable statement of what the detector *is*, and because
    ``EDetector`` — which does the same arithmetic on numpy arrays for speed — is checked
    against it observation by observation in the tests. When the two disagree, this one is
    right.
    """

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


class EDetector:
    """The detector used everywhere in benchlock: identical arithmetic, done on arrays.

    ``ReferenceEDetector`` keeps one Python object per change-point hypothesis, which costs
    roughly five seconds per 300-run stream at ``M=256`` — fine for a single CI run, and
    far too slow for the simulation study in Phase 5, which decides tens of thousands of
    streams. This class holds every candidate's state in parallel numpy arrays and updates
    them in one vectorised step.

    It is a performance rewrite and nothing more: the tests assert the two implementations
    agree observation by observation, and the conservative-pruning property (Hard Rule 4)
    holds here for the same structural reason — the retained statistic is a sub-sum of the
    full one over non-negative terms, with survivors' weights never redistributed.
    """

    __slots__ = (
        "_alarm_time",
        "_c",
        "_count",
        "_hi_up",
        "_lo_dn",
        "_log_e_cache",
        "_log_weight",
        "_logw_dn",
        "_logw_up",
        "_m2_dn",
        "_m2_up",
        "_mu0_dn",
        "_mu0_up",
        "_n_dn",
        "_n_pruned",
        "_n_up",
        "_peak_log_e",
        "_start",
        "_strategy_name",
        "_sum_dn",
        "_sum_up",
        "_t",
        "_theta",
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
        theta: float = 0.5,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if max_candidates is not None and max_candidates < 1:
            raise ValueError(f"max_candidates must be at least 1, got {max_candidates}")
        self.null = null
        self.alpha = alpha
        self.max_candidates = max_candidates
        self._c = c
        self._theta = theta
        self._strategy_name = strategy.name if strategy is not None else AgrapaBet().name
        if self._strategy_name not in ("agrapa", "predmix-eb"):
            raise ValueError(
                f"EDetector vectorises 'agrapa' and 'predmix-eb'; got {self._strategy_name!r}. "
                "Use ReferenceEDetector for other strategies"
            )

        self._mu0_up = null.least_favourable(Side.UP)
        self._mu0_dn = null.least_favourable(Side.DOWN)
        # Positivity bounds, narrowed to each leg's direction (see stats.betting.bounds).
        self._hi_up = c / self._mu0_up
        self._lo_dn = -c / (1.0 - self._mu0_dn)

        cap = 16 if max_candidates is None else min(max_candidates + 1, 1024)
        self._start = np.zeros(cap, dtype=np.int64)
        self._log_weight = np.zeros(cap, dtype=np.float64)
        self._logw_up = np.zeros(cap, dtype=np.float64)
        self._logw_dn = np.zeros(cap, dtype=np.float64)
        # Regularised running moments, matching WealthProcess's pseudo-observation at 1/2.
        self._sum_up = np.full(cap, 0.5, dtype=np.float64)
        self._sum_dn = np.full(cap, 0.5, dtype=np.float64)
        self._m2_up = np.full(cap, 0.25, dtype=np.float64)
        self._m2_dn = np.full(cap, 0.25, dtype=np.float64)
        self._n_up = np.ones(cap, dtype=np.float64)
        self._n_dn = np.ones(cap, dtype=np.float64)
        self._count = 0
        self._t = 0
        self._alarm_time: int | None = None
        self._n_pruned = 0
        #: The combined statistic is read several times per step (by `crossed`, by the
        #: caller, by the report). Recomputing a log-sum-exp each time dominated the
        #: profile, so it is computed once per update and cached until the next one.
        self._log_e_cache: float | None = None
        #: Running maximum of the combined statistic. Ville's inequality bounds
        #: ``P(exists t : E_t >= 1/alpha)`` — the supremum over time, not the value at the
        #: end. A drift that crossed and then subsided has still rejected the null, and
        #: reading only the endpoint would silently discard the anytime-valid property the
        #: whole tool is built on (and contradict `alarm_time`, which is already sticky).
        self._peak_log_e = -math.inf

    # ---- storage -----------------------------------------------------------------------

    def _grow(self) -> None:
        cap = self._start.size * 2
        for name in (
            "_start",
            "_log_weight",
            "_logw_up",
            "_logw_dn",
            "_sum_up",
            "_sum_dn",
            "_m2_up",
            "_m2_dn",
            "_n_up",
            "_n_dn",
        ):
            arr = getattr(self, name)
            grown = np.empty(cap, dtype=arr.dtype)
            grown[: arr.size] = arr
            setattr(self, name, grown)

    def _open_candidate(self) -> None:
        if self._count == self._start.size:
            self._grow()
        i = self._count
        self._start[i] = self._t
        self._log_weight[i] = changepoint_log_weight(self._t)
        self._logw_up[i] = 0.0
        self._logw_dn[i] = 0.0
        self._sum_up[i] = 0.5
        self._sum_dn[i] = 0.5
        self._m2_up[i] = 0.25
        self._m2_dn[i] = 0.25
        self._n_up[i] = 1.0
        self._n_dn[i] = 1.0
        self._count += 1

    # ---- the loop ----------------------------------------------------------------------

    def _bet(
        self,
        mean: NDArray[np.float64],
        var: NDArray[np.float64],
        n: NDArray[np.float64],
        mu0: float,
        lo: float,
        hi: float,
    ) -> NDArray[np.float64]:
        """Vectorised copy of the scalar betting strategies. Predictable by construction:
        every input here is a statistic of the history, computed before x is folded in."""
        if self._strategy_name == "agrapa":
            gap = mean - mu0
            lam = gap / (var + gap * gap)
        else:  # predmix-eb
            steps = n  # n = 1 + observations seen, which is the 1-based step index
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = np.sqrt(2.0 * math.log(1.0 / self.alpha) / (steps * np.log1p(steps) * var))
            lam = np.nan_to_num(lam, nan=0.0, posinf=hi if hi > 0 else lo)
            lam = np.minimum(lam, DEFAULT_TRUNCATION)
        return np.minimum(np.maximum(lam, lo), hi)

    def update(self, x: float) -> float:
        """Fold in one run's observation and return the combined log e-value."""
        if not 0.0 <= x <= 1.0:
            raise ValueError(
                f"observation {x} is outside [0, 1]. Betting e-processes require bounded "
                "observations; normalise with the declared score_scale before monitoring"
            )
        # A change could begin at this run, so open a hypothesis for it first.
        self._open_candidate()
        k = self._count

        for mu0, lo, hi, logw, total, m2, n in (
            (self._mu0_up, 0.0, self._hi_up, self._logw_up, self._sum_up, self._m2_up, self._n_up),
            (self._mu0_dn, self._lo_dn, 0.0, self._logw_dn, self._sum_dn, self._m2_dn, self._n_dn),
        ):
            mean = total[:k] / n[:k]
            var = m2[:k] / n[:k]
            lam = self._bet(mean, var, n[:k], mu0, lo, hi)
            logw[:k] += np.log1p(lam * (x - mu0))
            n[:k] += 1.0
            total[:k] += x
            m2[:k] += (x - total[:k] / n[:k]) ** 2

        self._t += 1
        self._prune()
        self._log_e_cache = self._compute_log_e()
        self._peak_log_e = max(self._peak_log_e, self._log_e_cache)
        if self._alarm_time is None and self._log_e_cache >= math.log(self.threshold):
            self._alarm_time = self._t - 1
        return self._log_e_cache

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
        if self.max_candidates is None or self._count <= self.max_candidates:
            return
        keep = np.argsort(self._log_contributions(), kind="stable")[::-1][: self.max_candidates]
        keep.sort()  # preserve chronological order so `start` stays readable
        self._n_pruned += self._count - self.max_candidates
        for name in (
            "_start",
            "_log_weight",
            "_logw_up",
            "_logw_dn",
            "_sum_up",
            "_sum_dn",
            "_m2_up",
            "_m2_dn",
            "_n_up",
            "_n_dn",
        ):
            arr = getattr(self, name)
            arr[: keep.size] = arr[keep]
        self._count = int(keep.size)

    # ---- the statistic -----------------------------------------------------------------

    def _log_e_each(self) -> NDArray[np.float64]:
        """Per-candidate log e-value: max of the weighted up and down legs."""
        k = self._count
        return np.maximum(
            math.log(self._theta) + self._logw_up[:k],
            math.log(1.0 - self._theta) + self._logw_dn[:k],
        )

    def _log_contributions(self) -> NDArray[np.float64]:
        return self._log_weight[: self._count] + self._log_e_each()

    def _compute_log_e(self) -> float:
        if self._count == 0:
            return -math.inf
        contributions = self._log_contributions()
        hi = float(contributions.max())
        if hi == -math.inf:  # pragma: no cover
            return -math.inf
        return hi + float(np.log(np.exp(contributions - hi).sum()))

    @property
    def log_e(self) -> float:
        if self._log_e_cache is None:
            self._log_e_cache = self._compute_log_e()
        return self._log_e_cache

    @property
    def e_value(self) -> float:
        try:
            return math.exp(self.log_e)
        except OverflowError:  # pragma: no cover
            return math.inf

    @property
    def peak_log_e(self) -> float:
        """Largest combined log e-value seen so far. This is what Ville's bound covers."""
        return max(self._peak_log_e, self.log_e) if self._count else -math.inf

    @property
    def peak_e_value(self) -> float:
        """The evidence the null has to answer for: the supremum over time, not the end."""
        try:
            return math.exp(self.peak_log_e)
        except OverflowError:  # pragma: no cover
            return math.inf

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha

    def crossed(self) -> bool:
        """Has the process EVER reached the threshold? Sticky, matching `alarm_time`."""
        return self.peak_log_e >= math.log(self.threshold)

    @property
    def alarm_time(self) -> int | None:
        """0-based run index at which the detector first crossed, or None."""
        return self._alarm_time

    @property
    def best_changepoint(self) -> int | None:
        """Where the change most likely began. **Descriptive only** — no verdict reads it.

        Ranked by each hypothesis's own evidence, not its weighted contribution: the
        weights are a prior that falls off steeply with j, so ranking by contribution
        would report the earliest candidate almost regardless of the data.
        """
        if self._count == 0:
            return None
        return int(self._start[: self._count][int(np.argmax(self._log_e_each()))])

    @property
    def direction(self) -> Side | None:
        if self._count == 0:
            return None
        best = int(np.argmax(self._log_e_each()))
        return Side.UP if self._logw_up[best] >= self._logw_dn[best] else Side.DOWN

    @property
    def n_candidates(self) -> int:
        return self._count

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
