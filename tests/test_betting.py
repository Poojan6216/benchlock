"""Phase 1.1 verify: the wealth process is positive, predictable, and a martingale.

These three properties are the whole guarantee. If any of them fails, every verdict this
tool produces is decoration.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchlock.stats.betting import (
    AgrapaBet,
    BetState,
    FixedBet,
    HedgedWealthProcess,
    Side,
    WealthProcess,
    make_strategy,
    wealth_trajectory,
)

STREAMS = st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=200)
MU0 = st.floats(0.05, 0.95, allow_nan=False)
STRATEGIES = st.sampled_from(["fixed", "agrapa"])
SIDES = st.sampled_from([Side.UP, Side.DOWN])
SLOW = settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# --- (a) wealth is always strictly positive -------------------------------------------------


@SLOW
@given(xs=STREAMS, mu0=MU0, name=STRATEGIES, side=SIDES)
def test_wealth_is_always_strictly_positive(
    xs: list[float], mu0: float, name: str, side: Side
) -> None:
    process = WealthProcess(mu0, side=side, strategy=make_strategy(name))
    for x in xs:
        process.update(x)
        assert math.isfinite(process.log_wealth), "log-wealth must stay finite"
        assert process.wealth > 0.0


@SLOW
@given(xs=STREAMS, mu0=MU0, name=STRATEGIES)
def test_every_multiplier_is_at_least_one_minus_c(xs: list[float], mu0: float, name: str) -> None:
    # The truncation exists to keep 1 + lambda(x - mu0) >= 1 - c. Check it directly on
    # the extreme observations, which are where positivity is at risk.
    process = WealthProcess(mu0, side=Side.UP, strategy=make_strategy(name))
    for x in xs:
        lam = process.next_bet()
        for extreme in (0.0, 1.0, x):
            assert 1.0 + lam * (extreme - mu0) >= 1.0 - process.c - 1e-12
        process.update(x)


@given(mu0=MU0, side=SIDES)
def test_bounds_respect_the_positivity_constraint(mu0: float, side: Side) -> None:
    lo, hi = WealthProcess(mu0, side=side).bounds
    # Positivity needs lambda in (-1/(1-mu0), 1/mu0); c=0.5 halves that range.
    assert hi <= 0.5 / mu0 + 1e-12
    assert lo >= -0.5 / (1.0 - mu0) - 1e-12


def test_mu0_at_the_boundary_is_refused() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="mu0 must be strictly inside"):
            WealthProcess(bad)


def test_observations_outside_the_unit_interval_are_refused() -> None:
    process = WealthProcess(0.5)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        process.update(1.5)


# --- (b) lambda_t is predictable: it never reads X_t ------------------------------------------


class PoisonedStrategy:
    """Records how much history was visible when each bet was chosen."""

    name = "poisoned"

    def __init__(self) -> None:
        self.inner = AgrapaBet()
        self.seen: list[int] = []

    def bet(self, state: BetState) -> float:
        self.seen.append(state.t)
        return self.inner.bet(state)


def test_bet_for_step_t_sees_exactly_t_minus_one_observations() -> None:
    spy = PoisonedStrategy()
    process = WealthProcess(0.5, strategy=spy)
    for x in [0.1, 0.9, 0.4, 0.6, 0.2]:
        process.update(x)
    # The bet for observation #1 is chosen having seen 0, and so on.
    assert spy.seen == [0, 1, 2, 3, 4]


@SLOW
@given(
    prefix=st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=0, max_size=60),
    a=st.floats(0.0, 1.0, allow_nan=False),
    b=st.floats(0.0, 1.0, allow_nan=False),
    mu0=MU0,
    name=STRATEGIES,
)
def test_the_bet_is_identical_on_streams_that_differ_only_at_the_current_step(
    prefix: list[float], a: float, b: float, mu0: float, name: str
) -> None:
    """The behavioural proof of predictability.

    Two streams sharing a prefix must produce the same bet for the next step, no matter
    how different the next observation is. If the bet could see X_t, this fails.
    """
    one = WealthProcess(mu0, strategy=make_strategy(name))
    two = WealthProcess(mu0, strategy=make_strategy(name))
    one.update_many(prefix)
    two.update_many(prefix)
    assert one.next_bet() == two.next_bet()

    # The wealth after the step differs only through the multiplier, since the bet that
    # produced it was fixed before either observation was seen.
    lam = one.next_bet()
    before = one.log_wealth
    one.update(a)
    two.update(b)
    assert one.log_wealth == pytest.approx(before + math.log(1 + lam * (a - mu0)), abs=1e-9)
    assert two.log_wealth == pytest.approx(before + math.log(1 + lam * (b - mu0)), abs=1e-9)


def test_a_poisoned_stream_would_be_detected() -> None:
    """Negative control: a strategy that peeks at X_t breaks the property test above.

    This proves the test has teeth rather than passing vacuously.
    """

    class Cheating:
        name = "cheating"

        def __init__(self) -> None:
            self.next_x = 0.0

        def bet(self, state: BetState) -> float:
            # A strategy that somehow saw the current observation would bet on it.
            return min(max(self.next_x - state.mu0, state.lo), state.hi)

    cheat = Cheating()
    one = WealthProcess(0.5, strategy=cheat)
    two = WealthProcess(0.5, strategy=cheat)
    one.update_many([0.5, 0.5])
    two.update_many([0.5, 0.5])
    cheat.next_x = 1.0
    first = one.next_bet()
    cheat.next_x = 0.0
    second = two.next_bet()
    assert first != second, "the cheating control must be visibly non-predictable"


# --- (c) under the null, E[K_t] <= 1 at every fixed t -----------------------------------------


@pytest.mark.parametrize("mu0", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("name", ["fixed", "agrapa"])
def test_wealth_is_a_martingale_under_the_null(mu0: float, name: str) -> None:
    """E[K_t] = 1 for all t under H0. Checked empirically over 2000 streams.

    Bernoulli(mu0) observations satisfy the null exactly, so any excess is either a
    broken martingale or Monte-Carlo noise — and the tolerance is the measured standard
    error, not a number chosen to make the test pass.
    """
    rng = np.random.default_rng(20260904)
    n_streams, horizon = 2000, 60
    xs = rng.binomial(1, mu0, size=(n_streams, horizon)).astype(float)

    log_wealth = np.zeros(n_streams)
    wealth_by_t = np.zeros((n_streams, horizon))
    processes = [
        WealthProcess(mu0, side=Side.UP, strategy=make_strategy(name)) for _ in range(n_streams)
    ]
    for t in range(horizon):
        for i, process in enumerate(processes):
            process.update(float(xs[i, t]))
            log_wealth[i] = process.log_wealth
        wealth_by_t[:, t] = np.exp(log_wealth)

    for t in range(horizon):
        column = wealth_by_t[:, t]
        mean = float(column.mean())
        se = float(column.std(ddof=1) / math.sqrt(n_streams))
        assert mean <= 1.0 + 3.0 * se + 1e-9, (
            f"E[K_{t + 1}] = {mean:.4f} exceeds 1 by more than 3 SE ({se:.4f}) "
            f"for mu0={mu0}, strategy={name}"
        )


def test_continuous_observations_also_satisfy_the_martingale_property() -> None:
    rng = np.random.default_rng(7)
    n_streams, horizon, mu0 = 2000, 40, 0.6
    xs = rng.beta(3.0, 2.0, size=(n_streams, horizon))  # mean = 0.6 exactly
    assert abs(xs.mean() - mu0) < 0.01

    finals = []
    for i in range(n_streams):
        process = HedgedWealthProcess(mu0)
        process.update_many(xs[i].tolist())
        finals.append(process.wealth)
    arr = np.array(finals)
    se = float(arr.std(ddof=1) / math.sqrt(n_streams))
    assert float(arr.mean()) <= 1.0 + 3.0 * se


def test_villes_inequality_holds_empirically() -> None:
    """The headline claim: P(sup_t K_t >= 1/alpha) <= alpha, under repeated looking."""
    rng = np.random.default_rng(4242)
    n_streams, horizon, mu0, alpha = 2000, 200, 0.5, 0.05
    xs = rng.binomial(1, mu0, size=(n_streams, horizon)).astype(float)

    alarms = 0
    for i in range(n_streams):
        process = HedgedWealthProcess(mu0)
        for t in range(horizon):
            process.update(float(xs[i, t]))
            if process.crossed(alpha):  # peeking at EVERY step, which is the point
                alarms += 1
                break
    rate = alarms / n_streams
    assert rate <= alpha, f"false-alarm rate {rate:.4f} exceeds alpha={alpha}"


# --- strategies and the hedged process ----------------------------------------------------------


def test_agrapa_bets_toward_the_observed_deviation() -> None:
    up = WealthProcess(0.5, side=Side.UP, strategy=AgrapaBet())
    up.update_many([0.9] * 10)
    assert up.next_bet() > 0.0, "after high observations, an up-process should bet positively"

    down = WealthProcess(0.5, side=Side.DOWN, strategy=AgrapaBet())
    down.update_many([0.1] * 10)
    assert down.next_bet() < 0.0


def test_agrapa_grows_wealth_against_a_false_null() -> None:
    rng = np.random.default_rng(1)
    xs = rng.binomial(1, 0.8, size=200).astype(float).tolist()
    agrapa = WealthProcess(0.5, side=Side.UP, strategy=AgrapaBet())
    agrapa.update_many(xs)
    assert agrapa.crossed(0.05), "agrapa must reject a badly false null within 200 samples"


def test_fixed_strategy_is_truncated_into_the_safe_range() -> None:
    process = WealthProcess(0.1, side=Side.UP, strategy=FixedBet(value=1000.0))
    _lo, hi = process.bounds
    assert process.next_bet() == hi
    assert hi == pytest.approx(0.5 / 0.1)


def test_hedged_process_detects_drift_in_either_direction() -> None:
    rng = np.random.default_rng(11)
    for true_mean in (0.2, 0.8):
        xs = rng.binomial(1, true_mean, size=300).astype(float).tolist()
        process = HedgedWealthProcess(0.5)
        process.update_many(xs)
        assert process.crossed(0.05), f"failed to detect a shift to {true_mean}"
    assert HedgedWealthProcess(0.5).direction in (Side.UP, Side.DOWN)


def test_hedged_wealth_is_the_mixture_of_its_legs() -> None:
    xs = [0.9, 0.8, 0.7, 0.95, 0.85]
    process = HedgedWealthProcess(0.5)
    process.update_many(xs)
    expected = 0.5 * math.exp(process.up.log_wealth) + 0.5 * math.exp(process.down.log_wealth)
    assert process.wealth == pytest.approx(expected, rel=1e-12)


def test_log_wealth_survives_a_long_extreme_run() -> None:
    # 5000 steps of maximal evidence: wealth would overflow a float, log-wealth does not.
    process = WealthProcess(0.5, side=Side.UP, strategy=FixedBet(1.0))
    process.update_many([1.0] * 5000)
    assert math.isfinite(process.log_wealth)
    assert process.log_wealth > 1000.0
    assert process.wealth == math.inf


def test_two_sided_side_is_rejected_by_the_one_sided_process() -> None:
    with pytest.raises(ValueError, match="one-sided"):
        WealthProcess(0.5, side=Side.TWO_SIDED)


def test_unknown_strategy_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown betting strategy"):
        make_strategy("kelly-ish")


def test_wealth_trajectory_matches_stepwise_updates() -> None:
    xs = [0.6, 0.7, 0.2, 0.9]
    traj = wealth_trajectory(xs, 0.5)
    process = HedgedWealthProcess(0.5)
    stepwise = []
    for x in xs:
        process.update(x)
        stepwise.append(process.log_wealth)
    assert traj == pytest.approx(stepwise)
