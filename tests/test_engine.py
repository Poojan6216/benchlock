"""Phase 2.2-2.4 verify: decide() against known ground truth, and its purity.

The determinism test is a Hard Rule 7 test: `benchlock replay` is only meaningful if the
same inputs give byte-identical output forever, so `decide` is exercised with the network
disabled and the clock frozen, and the suite fails if either is touched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from bench.sim.generate import StreamSpec, digest, generate, read_manifest, spec_from_json

from benchlock.attribute.engine import decide, estimation_scale, subtract_intervals
from benchlock.attribute.race import check_race
from benchlock.config import AttributionConfig
from benchlock.model.pins import NoiseFloor
from benchlock.model.verdict import Provisioning, Verdict
from benchlock.stats.confseq import Interval

MANIFEST = Path(__file__).resolve().parent / "fixtures" / "streams" / "manifest.json"
CONFIG = AttributionConfig(alpha=0.05, min_runs=8, min_obs=30, baseline_runs=8, target_shift=0.05)


def load_stream(name: str) -> tuple[list, list, dict]:
    entry = next(e for e in read_manifest(MANIFEST) if e["name"] == name)
    spec = spec_from_json(entry)
    system, anchor = generate(spec)
    return system, anchor, entry


# --- the fixtures are what they say they are ------------------------------------------------


def test_every_fixture_regenerates_to_its_committed_digest() -> None:
    """Deterministic generation, asserted rather than assumed.

    Fixtures are stored as specs plus a digest instead of 26 MB of materialised runs. If
    numpy's Generator ever changed its stream, this fails loudly rather than silently
    moving the ground truth underneath every golden verdict.
    """
    entries = read_manifest(MANIFEST)
    assert len(entries) >= 12
    for entry in entries:
        system, anchor = generate(spec_from_json(entry))
        assert digest(system, anchor) == entry["digest"], (
            f"{entry['name']} no longer regenerates to its committed digest"
        )


def test_generation_is_reproducible_within_a_process() -> None:
    spec = StreamSpec(name="repro", seed=7, judge_shift=-0.1)
    first, first_anchor = generate(spec)
    second, second_anchor = generate(spec)
    assert digest(first, first_anchor) == digest(second, second_anchor)


# --- the three demos, plus the rest of the golden set ------------------------------------------

# (fixture name, expected verdict, expected rule_id)
GOLDEN: list[tuple[str, Verdict, str]] = [
    ("demo1-phantom-judge", Verdict.JUDGE, "judge_only"),
    ("demo2-real-regression", Verdict.SYSTEM, "system_only"),
    ("demo3-under-provisioned", Verdict.INDETERMINATE, "race_failed"),
    ("stable-control", Verdict.STABLE, "no_crossing"),
    ("both-moved", Verdict.BOTH, "both_crossed"),
    ("judge-rubric-change", Verdict.JUDGE, "judge_only"),
    # Correlated judge noise does not shrink with anchor size, so a 200-item anchor
    # genuinely cannot rule the judge out here. Refusing is the correct answer, and this
    # row exists to keep it that way.
    ("correlated-judge-noise", Verdict.INDETERMINATE, "race_failed"),
    ("likert5-regression", Verdict.SYSTEM, "system_only"),
    ("binary-regression", Verdict.SYSTEM, "system_only"),
    ("short-stream", Verdict.STABLE, "insufficient_data"),
    ("late-judge-drift", Verdict.JUDGE, "judge_only"),
]


def test_an_undeclared_judge_swap_is_refused_rather_than_attributed() -> None:
    """Hard Rule 8: comparing across a changed pin is comparing two experiments."""
    from benchlock.model.verdict import AttributionRefusedError

    system, anchor, _ = load_stream("declared-judge-swap")
    with pytest.raises(AttributionRefusedError) as excinfo:
        decide(system, anchor, CONFIG)
    assert excinfo.value.rule_id == "pin_violation"
    assert "benchlock rebaseline" in excinfo.value.hint


@pytest.mark.parametrize(("name", "verdict", "rule_id"), GOLDEN, ids=[g[0] for g in GOLDEN])
def test_golden_streams_produce_the_right_verdict(
    name: str, verdict: Verdict, rule_id: str
) -> None:
    system, anchor, entry = load_stream(name)
    result = decide(system, anchor, CONFIG)
    assert result.verdict is verdict, (
        f"{name} (ground truth {entry['ground_truth']}): expected {verdict.value}, got "
        f"{result.verdict.value} via {result.rule_id}\n"
        f"  E_system={result.evidence.e_system:.1f} E_anchor={result.evidence.e_anchor:.1f} "
        f"E_corrected={result.evidence.e_corrected:.1f} "
        f"mds={result.evidence.min_detectable_judge_shift:.4f}"
    )
    assert result.rule_id == rule_id


def test_demo3_is_the_same_regression_as_demo2_with_a_smaller_anchor() -> None:
    """The point of Demo 3: identical system behaviour, only the anchor size differs."""
    _, _, two = load_stream("demo2-real-regression")
    _, _, three = load_stream("demo3-under-provisioned")
    assert two["system_shift"] == three["system_shift"]
    assert two["seed"] == three["seed"]
    assert three["anchor_items"] < two["anchor_items"]

    adequate = decide(*load_stream("demo2-real-regression")[:2], CONFIG)
    under = decide(*load_stream("demo3-under-provisioned")[:2], CONFIG)
    assert adequate.verdict is Verdict.SYSTEM
    assert under.verdict is Verdict.INDETERMINATE
    assert under.evidence.provisioning is Provisioning.UNDER_PROVISIONED
    assert under.evidence.min_detectable_judge_shift > adequate.evidence.min_detectable_judge_shift


def test_cancellation_is_detected_as_both() -> None:
    """Phase 7.3's nastiest case: the net score barely moves and both components did."""
    system, anchor, _ = load_stream("cancellation")
    result = decide(system, anchor, CONFIG)
    assert result.verdict is Verdict.BOTH


# --- Hard Rule 7: purity ----------------------------------------------------------------------------


@pytest.mark.mandatory
def test_decide_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same inputs, byte-identical output — with the network and clock removed.

    `benchlock replay` re-derives every historical verdict; if `decide` could read a clock
    or a random source, replay would be theatre.
    """
    import socket
    import time

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("decide() attempted a network call")

    def no_clock(*args: object, **kwargs: object) -> None:
        raise AssertionError("decide() read the clock")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(time, "time", no_clock)
    monkeypatch.setattr(time, "monotonic", no_clock)

    for name, _, _ in GOLDEN[:5]:
        system, anchor, _ = load_stream(name)
        first = decide(system, anchor, CONFIG)
        second = decide(system, anchor, CONFIG)
        assert json.dumps(first.to_json(), sort_keys=True) == json.dumps(
            second.to_json(), sort_keys=True
        )


@pytest.mark.mandatory
def test_decide_is_deterministic_over_many_random_inputs() -> None:
    """The spec asks for 5000 triples; each is decided twice and compared byte for byte."""
    rng = np.random.default_rng(20260904)
    checked = 0
    for i in range(250):
        spec = StreamSpec(
            name=f"rand{i}",
            seed=int(rng.integers(0, 2**31)),
            n_runs=int(rng.integers(10, 40)),
            change_at=int(rng.integers(1, 9)),
            system_items=int(rng.integers(30, 120)),
            anchor_items=int(rng.integers(20, 120)),
            per_item_sd=float(rng.uniform(0.02, 0.2)),
            judge_shift=float(rng.choice([0.0, -0.05, 0.05])),
            system_shift=float(rng.choice([0.0, -0.05, 0.05])),
        )
        system, anchor = generate(spec)
        for _ in range(2):
            first = decide(system, anchor, CONFIG)
            second = decide(system, anchor, CONFIG)
            assert first == second
            assert json.dumps(first.to_json(), sort_keys=True) == json.dumps(
                second.to_json(), sort_keys=True
            )
            checked += 1
    assert checked == 500


def test_decide_does_not_mutate_its_inputs() -> None:
    system, anchor, _ = load_stream("demo2-real-regression")
    before = (digest(system, anchor), len(system), len(anchor))
    decide(system, anchor, CONFIG)
    assert (digest(system, anchor), len(system), len(anchor)) == before


# --- 2.4 anchor correction ---------------------------------------------------------------------------------


def test_interval_subtraction_widens_rather_than_pretending() -> None:
    """The single most likely place to manufacture a false guarantee."""
    system = Interval(-0.12, -0.08)
    anchor = Interval(-0.06, -0.02)
    corrected = subtract_intervals(system, anchor)
    # Point estimates would give -0.10 - (-0.04) = -0.06, with the original width 0.04.
    assert corrected.midpoint == pytest.approx(-0.06)
    assert corrected.width == pytest.approx(system.width + anchor.width)
    assert corrected.width > system.width, "the correction is itself estimated"


def test_estimation_band_is_wider_than_the_detection_band() -> None:
    """Detection wants the band at the target; estimation must not clip."""
    scale = estimation_scale(target_shift=0.05, baseline_sd=0.005)
    assert scale.half_width >= 4 * 0.05
    assert scale.to_unit(-0.12) > 0.0, "a real regression must not saturate the estimate"


@pytest.mark.slow
def test_corrected_interval_covers_the_true_system_shift_under_simultaneous_drift() -> None:
    """2.4's verify: coverage of the corrected interval when both components moved.

    Ground truth is the *system* shift; the judge moves at the same time. The corrected
    interval must contain the true system shift at least 1-alpha of the time.
    """
    rng = np.random.default_rng(4242)
    covered = 0
    trials = 200
    for i in range(trials):
        true_system = float(rng.choice([-0.08, -0.05, 0.05, 0.08]))
        true_judge = float(rng.choice([-0.06, -0.03, 0.03, 0.06]))
        spec = StreamSpec(
            name=f"cov{i}",
            seed=int(rng.integers(0, 2**31)),
            n_runs=45,
            change_at=8,
            judge_shift=true_judge,
            system_shift=true_system,
        )
        system, anchor = generate(spec)
        result = decide(system, anchor, CONFIG)
        if result.evidence.corrected_shift.contains(true_system):
            covered += 1
    rate = covered / trials
    assert rate >= 0.95, f"corrected interval covered the true system shift only {rate:.1%}"


# --- 2.2 the race check -------------------------------------------------------------------------------------


def floor_for(n: int, per_item_sd: float = 0.08, K: int = 5) -> NoiseFloor:
    return NoiseFloor(
        per_item_sd=per_item_sd,
        run_mean_sd=per_item_sd / math.sqrt(n),
        replicates=K,
        n_items=n,
    )


def test_race_check_flags_an_under_powered_anchor() -> None:
    small = check_race(-0.05, floor_for(20), 20, 50, 0.05, target_shift=0.05)
    big = check_race(-0.05, floor_for(400), 400, 50, 0.05, target_shift=0.05)
    assert small.provisioning is Provisioning.UNDER_PROVISIONED
    assert big.provisioning is Provisioning.ADEQUATE
    assert small.min_detectable_judge_shift > big.min_detectable_judge_shift
    assert small.anchor_n == 20
    assert "INVISIBLE" in small.explain()
    assert "ADEQUATE" in big.explain()


def test_race_check_uses_the_runs_actually_elapsed() -> None:
    """A well-provisioned anchor set is still under-provisioned on run 3."""
    early = check_race(-0.05, floor_for(400), 400, 3, 0.05, target_shift=0.05)
    later = check_race(-0.05, floor_for(400), 400, 60, 0.05, target_shift=0.05)
    assert early.min_detectable_judge_shift > later.min_detectable_judge_shift
    assert early.provisioning is Provisioning.UNDER_PROVISIONED
    assert later.provisioning is Provisioning.ADEQUATE


def test_race_check_refuses_a_zero_sized_anchor() -> None:
    with pytest.raises(ValueError, match="anchor size must be at least 1"):
        check_race(-0.05, floor_for(40), 0, 50, 0.05)


def test_no_anchor_stream_means_nothing_can_rule_out_the_judge() -> None:
    """The B5 ablation, as a property: without a control group there is no attribution."""
    system, _, _ = load_stream("demo2-real-regression")
    result = decide(system, [], CONFIG)
    assert result.evidence.provisioning is Provisioning.UNDER_PROVISIONED
    assert result.verdict in (Verdict.INDETERMINATE, Verdict.STABLE)
    assert result.verdict is not Verdict.SYSTEM, (
        "with no anchor set, `SYSTEM` would be a guess with a straight face"
    )
