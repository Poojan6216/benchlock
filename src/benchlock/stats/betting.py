"""Betting wealth processes: the primitive underneath every guarantee in this tool.

To test ``H0: E[X_t] = mu0`` for observations bounded in [0, 1], bet against the null and
track your wealth::

    K_0 = 1,    K_t = K_{t-1} * (1 + lambda_t * (X_t - mu0))

If ``lambda_t`` is **predictable** — a function of ``X_1..X_{t-1}`` only — then under the
null ``E[K_t | F_{t-1}] = K_{t-1}``, so ``(K_t)`` is a non-negative martingale starting at
1. Ville's inequality (1939) then gives, for *all* t simultaneously::

    P(exists t : K_t >= 1/alpha)  <=  alpha

That single line is why this tool can be looked at every day without inflating its
false-alarm rate. Everything else in `stats/` is built on it.

Two implementation details carry real weight:

* **Predictability is structural.** ``update()`` computes the bet from state that has not
  yet seen the current observation, then folds the observation in. A strategy is handed a
  `BetState` and physically cannot read ``X_t``. The property tests verify this
  behaviourally as well, by feeding two streams that differ only at position t.
* **Wealth is tracked in logs.** ``1 + lambda(X - mu0) >= 1 - c > 0`` by truncation, so the
  log is always finite, and a process that runs for thousands of steps neither overflows
  nor underflows to zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

#: Truncation factor. Keeps every wealth multiplier at or above ``1 - c``, so wealth stays
#: strictly positive and the log never blows up. 0.5 is the spec's choice.
DEFAULT_TRUNCATION = 0.5


class Side(StrEnum):
    """Which alternative the process bets on."""

    UP = "up"  # the mean rose above mu0
    DOWN = "down"  # the mean fell below mu0
    TWO_SIDED = "two-sided"  # hedged 50/50 mixture of the two; the default for drift


@dataclass(frozen=True, slots=True)
class BetState:
    """Everything the strategy is allowed to know when choosing ``lambda_t``.

    It carries statistics of ``X_1..X_{t-1}`` and nothing about ``X_t``. This type *is*
    the predictability guarantee: a strategy has no channel through which to cheat.
    """

    t: int  # observations seen so far (so the bet being chosen is for step t+1)
    mean: float  # regularised running mean of the history
    var: float  # regularised running variance of the history
    mu0: float  # the null mean under test
    lo: float  # lower truncation bound for lambda
    hi: float  # upper truncation bound for lambda


class BettingStrategy(Protocol):
    """Chooses the next bet. Must be a pure function of `BetState`."""

    @property
    def name(self) -> str: ...

    def bet(self, state: BetState) -> float: ...


@dataclass(frozen=True, slots=True)
class FixedBet:
    """A constant bet, truncated into the safe range. Used to make tests legible."""

    value: float = 0.5
    name: str = "fixed"

    def bet(self, state: BetState) -> float:
        return min(max(self.value, state.lo), state.hi)


@dataclass(frozen=True, slots=True)
class AgrapaBet:
    """Approximate-GRAPA (Waudby-Smith & Ramdas): the plug-in growth-rate-optimal bet.

    ``lambda_t = (mu_hat - mu0) / (var_hat + (mu_hat - mu0)^2)`` evaluated on the history,
    which is the value maximising expected log-wealth against the alternative the data so
    far suggests. Predictable by construction, since the estimates are of the history only.
    """

    name: str = "agrapa"

    def bet(self, state: BetState) -> float:
        gap = state.mean - state.mu0
        denom = state.var + gap * gap
        if denom <= 0.0:  # pragma: no cover - the regularised variance is always > 0
            return 0.0
        return min(max(gap / denom, state.lo), state.hi)


def make_strategy(name: str) -> BettingStrategy:
    if name == "fixed":
        return FixedBet()
    if name == "agrapa":
        return AgrapaBet()
    raise ValueError(f"unknown betting strategy {name!r}; use 'fixed' or 'agrapa'")


class WealthProcess:
    """A one-sided betting martingale for ``H0: E[X_t] = mu0``.

    Wealth is exposed as ``log_wealth`` (always finite) and ``wealth`` (may be inf for a
    process that has run away from the null, which is exactly the case where the precise
    value stops mattering).
    """

    __slots__ = ("_m2", "_n", "_sum", "c", "log_wealth", "mu0", "side", "strategy", "t")

    def __init__(
        self,
        mu0: float,
        *,
        side: Side = Side.UP,
        strategy: BettingStrategy | None = None,
        c: float = DEFAULT_TRUNCATION,
    ) -> None:
        if not 0.0 < mu0 < 1.0:
            raise ValueError(
                f"mu0 must be strictly inside (0, 1), got {mu0}. A null mean at the boundary "
                "admits no safe bet: the wealth multiplier could hit zero"
            )
        if not 0.0 < c < 1.0:
            raise ValueError(f"truncation c must be in (0, 1), got {c}")
        if side is Side.TWO_SIDED:
            raise ValueError("WealthProcess is one-sided; use HedgedWealthProcess for two-sided")
        self.mu0 = mu0
        self.side = side
        self.c = c
        self.strategy: BettingStrategy = strategy if strategy is not None else AgrapaBet()
        self.log_wealth = 0.0
        self.t = 0
        # Regularised running moments: a pseudo-observation at 1/2 with weight 1, which
        # keeps the variance strictly positive from the very first step.
        self._sum = 0.5
        self._m2 = 0.25
        self._n = 1.0

    # ---- bounds -------------------------------------------------------------------------

    @property
    def bounds(self) -> tuple[float, float]:
        """The safe truncation range for lambda, narrowed to the side being tested.

        Positivity needs ``lambda in (-1/(1-mu0), 1/mu0)``; scaling by ``c`` keeps every
        multiplier at or above ``1 - c``.
        """
        upper = self.c / self.mu0
        lower = -self.c / (1.0 - self.mu0)
        if self.side is Side.UP:
            return 0.0, upper
        return lower, 0.0

    def _state(self) -> BetState:
        mean = self._sum / self._n
        var = self._m2 / self._n
        lo, hi = self.bounds
        return BetState(t=self.t, mean=mean, var=var, mu0=self.mu0, lo=lo, hi=hi)

    # ---- the process --------------------------------------------------------------------

    def next_bet(self) -> float:
        """The bet that would be placed on the next observation. Predictable by design."""
        return self.strategy.bet(self._state())

    def update(self, x: float) -> float:
        """Fold in one observation and return the new log-wealth."""
        if not 0.0 <= x <= 1.0:
            raise ValueError(
                f"observation {x} is outside [0, 1]. Betting e-processes require bounded "
                "observations; normalise with the declared score_scale before monitoring"
            )
        lam = self.next_bet()  # chosen from the history only — never from x
        multiplier = 1.0 + lam * (x - self.mu0)
        # Guaranteed >= 1 - c > 0 by the truncation; assert rather than clamp, because a
        # violation here would mean the bounds are wrong and silence would hide it.
        if multiplier <= 0.0:  # pragma: no cover - unreachable while bounds hold
            raise AssertionError(
                f"non-positive wealth multiplier {multiplier} (lambda={lam}, x={x}, mu0={self.mu0})"
            )
        self.log_wealth += math.log(multiplier)

        self._n += 1.0
        self._sum += x
        prev_mean = self._sum / self._n
        self._m2 += (x - prev_mean) ** 2
        self.t += 1
        return self.log_wealth

    def update_many(self, xs: Iterable[float]) -> float:
        for x in xs:
            self.update(x)
        return self.log_wealth

    @property
    def wealth(self) -> float:
        try:
            return math.exp(self.log_wealth)
        except OverflowError:  # pragma: no cover - only for wildly rejected nulls
            return math.inf

    def crossed(self, alpha: float) -> bool:
        """Has wealth reached the 1/alpha threshold? This is the rejection rule."""
        return self.log_wealth >= math.log(1.0 / alpha)


class HedgedWealthProcess:
    """Two-sided drift detection: a 50/50 mixture of an up-betting and a down-betting process.

    A convex combination of martingales is a martingale, so the mixture inherits Ville's
    inequality exactly. This is what the detector uses, because a judge or a system can
    drift in either direction and we must not have to guess which in advance.
    """

    __slots__ = ("down", "theta", "up")

    def __init__(
        self,
        mu0: float,
        *,
        strategy: BettingStrategy | None = None,
        c: float = DEFAULT_TRUNCATION,
        theta: float = 0.5,
    ) -> None:
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"mixture weight theta must be in [0, 1], got {theta}")
        self.theta = theta
        self.up = WealthProcess(mu0, side=Side.UP, strategy=strategy, c=c)
        self.down = WealthProcess(mu0, side=Side.DOWN, strategy=strategy, c=c)

    @property
    def mu0(self) -> float:
        return self.up.mu0

    @property
    def t(self) -> int:
        return self.up.t

    def update(self, x: float) -> float:
        self.up.update(x)
        self.down.update(x)
        return self.log_wealth

    def update_many(self, xs: Iterable[float]) -> float:
        for x in xs:
            self.update(x)
        return self.log_wealth

    @property
    def log_wealth(self) -> float:
        """log(theta*K_up + (1-theta)*K_down), by log-sum-exp so it never overflows."""
        return _log_mix(self.theta, self.up.log_wealth, 1.0 - self.theta, self.down.log_wealth)

    @property
    def wealth(self) -> float:
        try:
            return math.exp(self.log_wealth)
        except OverflowError:  # pragma: no cover
            return math.inf

    def crossed(self, alpha: float) -> bool:
        return self.log_wealth >= math.log(1.0 / alpha)

    @property
    def direction(self) -> Side:
        """Which side currently holds more wealth — reported, never used to decide."""
        return Side.UP if self.up.log_wealth >= self.down.log_wealth else Side.DOWN


def _log_mix(w1: float, log_a: float, w2: float, log_b: float) -> float:
    """log(w1*exp(log_a) + w2*exp(log_b)), stable for extreme values."""
    if w1 <= 0.0:
        return math.log(w2) + log_b
    if w2 <= 0.0:
        return math.log(w1) + log_a
    hi = max(log_a, log_b)
    return hi + math.log(w1 * math.exp(log_a - hi) + w2 * math.exp(log_b - hi))


def wealth_trajectory(
    xs: Sequence[float],
    mu0: float,
    *,
    strategy: BettingStrategy | None = None,
    c: float = DEFAULT_TRUNCATION,
) -> list[float]:
    """Log-wealth of the two-sided process after each observation. For plots and tests."""
    process = HedgedWealthProcess(mu0, strategy=strategy, c=c)
    out = []
    for x in xs:
        process.update(x)
        out.append(process.log_wealth)
    return out
