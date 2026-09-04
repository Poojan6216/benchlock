"""Phase 1.3 verify: our confidence sequences against `confseq`, and against reality.

Two independent checks, because they catch different things:

1. **Differential** against the reference implementation. Catches construction errors —
   and did: an asymmetric pre-truncation of the bets that made the upper leg too weak.
2. **Coverage simulation.** Catches the error the differential cannot: both
   implementations being wrong in the same way. A confidence sequence's only real claim
   is that it covers the truth at every time simultaneously, so we measure that directly.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchlock.stats.confseq import (
    Interval,
    empirical_bernstein_cs,
    final_interval,
    hedged_capital,
    hedged_cs,
    mean_cs,
)

confseq = pytest.importorskip(
    "confseq.betting",
    reason="the `confseq` oracle needs Boost to build: `brew install boost && uv sync --group oracle`",
)


@pytest.fixture(scope="module")
def reference():
    """The reference implementations. The NumPy 2 shim lives in conftest."""
    from confseq.betting import hedged_cs as ref_hedged
    from confseq.predmix import predmix_empbern_twosided_cs as ref_eb

    return ref_hedged, ref_eb


def sample_streams(rng: np.random.Generator, n_streams: int, horizon: int) -> list:
    """A spread of shapes a real rubric might produce, including nasty ones."""
    out = []
    for i in range(n_streams):
        kind = i % 6
        if kind == 0:
            xs = rng.binomial(1, rng.uniform(0.1, 0.9), horizon).astype(float)
        elif kind == 1:
            xs = rng.beta(rng.uniform(0.5, 5), rng.uniform(0.5, 5), horizon)
        elif kind == 2:
            xs = rng.uniform(0, 1, horizon)
        elif kind == 3:  # near-ceiling: 99% at the top, the Phase 7.7 shape
            xs = np.clip(rng.normal(0.97, 0.02, horizon), 0, 1)
        elif kind == 4:  # Likert-5 normalised onto [0,1]
            xs = rng.integers(0, 5, horizon).astype(float) / 4.0
        else:  # zero variance
            xs = np.full(horizon, float(rng.uniform(0.1, 0.9)))
        out.append(xs)
    return out


# --- differential against the reference --------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_hedged_cs_matches_the_reference(reference, alpha: float) -> None:
    ref_hedged, _ = reference
    rng = np.random.default_rng(20260904)
    for xs in sample_streams(rng, 60, 40):
        rl, ru = ref_hedged(xs, alpha=alpha, breaks=200, running_intersection=True)
        ours = hedged_cs(xs, alpha, breaks=200)
        for t, interval in enumerate(ours):
            assert interval.lower == pytest.approx(rl[t], abs=1e-9)
            assert interval.upper == pytest.approx(ru[t], abs=1e-9)


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_empirical_bernstein_cs_matches_the_reference(reference, alpha: float) -> None:
    _, ref_eb = reference
    rng = np.random.default_rng(11)
    for xs in sample_streams(rng, 60, 40):
        el, eu = ref_eb(xs, alpha=alpha, running_intersection=True)
        ours = empirical_bernstein_cs(xs, alpha)
        for t, interval in enumerate(ours):
            assert interval.lower == pytest.approx(el[t], abs=1e-9)
            assert interval.upper == pytest.approx(eu[t], abs=1e-9)


def test_we_are_never_narrower_than_the_reference(reference) -> None:
    """The spec's rule: narrower than the reference is a bug, not an improvement.

    Runs the full 500-stream sweep and records the width-ratio distribution, which is
    reported in docs/statistical-guarantees.md.
    """
    ref_hedged, ref_eb = reference
    rng = np.random.default_rng(31337)
    ratios_hedged: list[float] = []
    ratios_eb: list[float] = []

    for xs in sample_streams(rng, 500, 25):
        rl, ru = ref_hedged(xs, alpha=0.05, breaks=100, running_intersection=True)
        for t, interval in enumerate(hedged_cs(xs, 0.05, breaks=100)):
            ref_width = float(ru[t] - rl[t])
            assert interval.width >= ref_width - 1e-9, (
                f"hedged CS narrower than the reference at t={t}: "
                f"{interval.width:.6f} < {ref_width:.6f}"
            )
            if ref_width > 1e-9:
                ratios_hedged.append(interval.width / ref_width)

        el, eu = ref_eb(xs, alpha=0.05, running_intersection=True)
        for t, interval in enumerate(empirical_bernstein_cs(xs, 0.05)):
            ref_width = float(eu[t] - el[t])
            assert interval.width >= ref_width - 1e-9, (
                f"EB CS narrower than the reference at t={t}: "
                f"{interval.width:.6f} < {ref_width:.6f}"
            )
            if ref_width > 1e-9:
                ratios_eb.append(interval.width / ref_width)

    summary = {
        "streams": 500,
        "horizon": 25,
        "alpha": 0.05,
        "hedged": _describe(ratios_hedged),
        "empirical_bernstein": _describe(ratios_eb),
    }
    out = Path(__file__).resolve().parent.parent / "bench" / "results" / "confseq-differential.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    # Exact agreement is what we expect, having matched the construction.
    assert summary["hedged"]["max"] == pytest.approx(1.0, abs=1e-6)
    assert summary["empirical_bernstein"]["max"] == pytest.approx(1.0, abs=1e-6)


def _describe(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


# --- coverage: the property that actually matters -------------------------------------------


@pytest.mark.parametrize("method", ["hedged", "empirical-bernstein"])
def test_time_uniform_coverage_holds(method: str) -> None:
    """P(for all t: mu in CS_t) >= 1 - alpha, measured rather than asserted.

    The differential test cannot catch both implementations being wrong together; this
    can. Coverage is checked over *every* time step, which is the quantifier that makes
    a confidence sequence different from a confidence interval.
    """
    rng = np.random.default_rng(9001)
    alpha, n_streams, horizon = 0.1, 400, 30
    misses = 0
    for _ in range(n_streams):
        mu = float(rng.uniform(0.2, 0.8))
        xs = rng.binomial(1, mu, horizon).astype(float)
        breaks = 200 if method == "hedged" else 1000
        seq = mean_cs(xs, alpha, method=method, breaks=breaks)
        if any(not interval.contains(mu) for interval in seq):
            misses += 1
    rate = misses / n_streams
    # Binomial slack at 400 streams; the point is that it is nowhere near alpha.
    assert rate <= alpha, f"{method}: time-uniform miscoverage {rate:.4f} exceeds alpha={alpha}"


def test_intervals_shrink_as_evidence_accumulates() -> None:
    rng = np.random.default_rng(5)
    xs = rng.binomial(1, 0.5, 400).astype(float)
    seq = hedged_cs(xs, 0.05, breaks=200)
    assert seq[-1].width < seq[9].width, "a CS must tighten with more data"
    assert seq[-1].contains(0.5)


def test_running_intersection_is_monotone() -> None:
    rng = np.random.default_rng(6)
    xs = rng.beta(2, 2, 120)
    seq = hedged_cs(xs, 0.05, breaks=200, running_intersection=True)
    for earlier, later in itertools.pairwise(seq):
        assert later.lower >= earlier.lower - 1e-12
        assert later.upper <= earlier.upper + 1e-12


# --- the Interval type ------------------------------------------------------------------------


def test_interval_basics() -> None:
    i = Interval(0.2, 0.6)
    assert i.width == pytest.approx(0.4)
    assert i.midpoint == pytest.approx(0.4)
    assert i.contains(0.2) and i.contains(0.6) and not i.contains(0.61)
    shifted = i.shifted(-0.1)
    assert (shifted.lower, shifted.upper) == pytest.approx((0.1, 0.5))
    assert i.intersect(Interval(0.4, 0.9)) == Interval(0.4, 0.6)
    assert Interval(-0.1, 0.2).contains(0.0)


def test_interval_rejects_inversion() -> None:
    with pytest.raises(ValueError, match="inverted"):
        Interval(0.7, 0.3)


def test_shift_interval_excluding_zero() -> None:
    assert Interval(-0.15, -0.05).excludes_zero_shift()
    assert not Interval(-0.05, 0.05).excludes_zero_shift()


# --- input validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("xs", "match"),
    [
        ([], "at least one observation"),
        ([0.5, 1.5], r"\[0, 1\]"),
        ([0.5, -0.1], r"\[0, 1\]"),
        ([0.5, float("nan")], "finite"),
    ],
)
def test_bad_observations_are_refused(xs: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        hedged_cs(xs, 0.05, breaks=50)


def test_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown CS method"):
        mean_cs([0.5, 0.5], 0.05, method="bootstrap")


def test_degenerate_null_means_give_infinite_capital() -> None:
    assert math.isinf(hedged_capital([0.5, 0.5], 0.0)[0])
    assert math.isinf(hedged_capital([0.5, 0.5], 1.0)[0])


def test_final_interval_is_the_last_of_the_sequence() -> None:
    xs = np.random.default_rng(2).beta(2, 3, 50)
    assert final_interval(xs, 0.05) == hedged_cs(xs, 0.05)[-1]
