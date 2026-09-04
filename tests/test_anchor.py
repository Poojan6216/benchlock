"""Phase 3 verify: anchor modes, selection, coverage, and the zero-label adoption path."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchlock.anchor.coverage import COVERAGE_WARN_THRESHOLD, measure_coverage
from benchlock.anchor.modes import (
    AnchorItem,
    AnchorModeError,
    agreement_series,
    check_mode_supported,
    freeze,
    load_anchors,
    rescore,
    save_anchors,
)
from benchlock.anchor.select import Candidate, decile_of, select_anchors
from benchlock.config import SelectionKind
from benchlock.judge.base import SimulatedJudge
from benchlock.model.pins import AnchorMode
from benchlock.stats.edetector import EDetector
from benchlock.stats.eprocess import MonitorScale, frozen_baseline_null, split_alpha

ALPHA = 0.05
_, ALPHA_M = split_alpha(ALPHA)


def anchor_items(n: int = 200, *, gold: bool = False, tags: bool = True) -> list[AnchorItem]:
    kinds = ["prose", "code", "summary", "qa"]
    return [
        AnchorItem(
            item_id=f"a{i}",
            prompt_input=f"question {i}",
            output=f"answer {i}",
            tags=(kinds[i % 4],) if tags else (),
            gold=(0.5 + 0.001 * i) if gold else None,
        )
        for i in range(n)
    ]


# --- 3.1 frozen-self: no labels anywhere -----------------------------------------------------


def test_freeze_needs_no_labels() -> None:
    frozen = freeze(anchor_items(50), SimulatedJudge(seed=1), replicates=5)
    assert frozen.mode is AnchorMode.FROZEN_SELF
    assert frozen.pin.n == 50
    assert len(frozen.baseline_scores) == 50
    assert frozen.noise.floor.replicates == 5
    assert all(0.0 <= s <= 1.0 for s in frozen.baseline_scores.values())
    assert len(frozen.replicate_scorings) == 5


def test_freeze_then_drift_the_judge_then_detect() -> None:
    """3.1's end-to-end verify: freeze, drift the judge, detect."""
    items = anchor_items(260)
    frozen = freeze(items, SimulatedJudge(seed=1), replicates=5)
    floor = frozen.noise.floor
    scale = MonitorScale.for_target(0.05, floor.run_mean_sd)
    null = frozen_baseline_null(floor.run_mean_sd, 5, ALPHA, scale=scale)

    detector = EDetector(null, ALPHA_M, max_candidates=64)
    for run in range(60):
        drift = -0.08 if run >= 10 else 0.0
        judge = SimulatedJudge(seed=100 + run, drift=drift)
        scores = rescore(items, judge)
        deviation = sum(scores[i.item_id] - frozen.baseline_scores[i.item_id] for i in items) / len(
            items
        )
        detector.update(scale.to_unit(deviation))
    assert detector.crossed(), "a 0.08 judge drift over 50 runs must be detected"
    assert detector.alarm_time is not None


def test_a_biased_but_stable_judge_reads_as_stable() -> None:
    """The limitation, stated as a test: stability is not validity.

    A judge that is consistently wrong stays consistently wrong. Benchlock must not
    confuse "wrong" with "drifting" — it makes no claim that your evals are correct.
    """
    items = anchor_items(260)
    frozen = freeze(items, SimulatedJudge(seed=2, bias=0.15), replicates=5)
    floor = frozen.noise.floor
    scale = MonitorScale.for_target(0.05, floor.run_mean_sd)
    null = frozen_baseline_null(floor.run_mean_sd, 5, ALPHA, scale=scale)

    detector = EDetector(null, ALPHA_M, max_candidates=64)
    for run in range(200):
        judge = SimulatedJudge(seed=500 + run, bias=0.15)  # wrong, but wrong the same way
        scores = rescore(items, judge)
        deviation = sum(scores[i.item_id] - frozen.baseline_scores[i.item_id] for i in items) / len(
            items
        )
        detector.update(scale.to_unit(deviation))
        assert not detector.crossed(), (
            f"a biased but stable judge was reported as drifting at run {run}"
        )


def test_freezing_an_empty_set_is_refused() -> None:
    with pytest.raises(AnchorModeError, match="cannot freeze an empty anchor set"):
        freeze([], SimulatedJudge())


def test_one_replicate_cannot_measure_a_floor() -> None:
    with pytest.raises(AnchorModeError, match="at least K=2 replicates"):
        freeze(anchor_items(10), SimulatedJudge(), replicates=1)


def test_a_judge_returning_out_of_range_scores_fails_loudly() -> None:
    judge = SimulatedJudge(scale=(1.0, 5.0))
    judge.scale = (1.0, 5.0)

    class OutOfRange(SimulatedJudge):
        def score(self, requests):  # type: ignore[no-untyped-def]
            from benchlock.judge.base import JudgeResult

            return [JudgeResult(r.item_id, 9.0) for r in requests]

    with pytest.raises(AnchorModeError, match="outside the declared score_scale"):
        freeze(anchor_items(5), OutOfRange(), replicates=2)


# --- 3.2 human mode ---------------------------------------------------------------------------


def test_human_mode_requires_labels_and_says_so() -> None:
    with pytest.raises(AnchorModeError, match="needs a gold label"):
        freeze(anchor_items(10), SimulatedJudge(), replicates=2, mode=AnchorMode.HUMAN)


def test_human_mode_adds_an_agreement_series_and_nothing_else() -> None:
    items = anchor_items(40, gold=True)
    frozen = freeze(items, SimulatedJudge(seed=3), replicates=3, mode=AnchorMode.HUMAN)
    assert frozen.mode is AnchorMode.HUMAN

    disagreement, n_labelled = agreement_series(frozen.baseline_scores, items)
    assert n_labelled == 40
    assert 0.0 <= disagreement <= 1.0

    # Without labels the same items produce an identical floor: the labels are additive.
    unlabelled = [AnchorItem(i.item_id, i.prompt_input, i.output, i.tags, gold=None) for i in items]
    plain = freeze(unlabelled, SimulatedJudge(seed=3), replicates=3)
    assert plain.pin.item_set_hash == frozen.pin.item_set_hash
    assert plain.noise.floor == frozen.noise.floor
    assert agreement_series(plain.baseline_scores, unlabelled) == (0.0, 0)


# --- 3.3 replicate mode -----------------------------------------------------------------------


def test_replicate_mode_is_refused_without_pinned_snapshots() -> None:
    judge = SimulatedJudge(supports_pinned_snapshots=False, model="rolling-latest")
    with pytest.raises(AnchorModeError) as excinfo:
        check_mode_supported(AnchorMode.REPLICATE, judge)
    assert "pinned dated snapshots" in excinfo.value.message
    assert "frozen-self" in excinfo.value.hint


def test_replicate_mode_is_accepted_with_pinned_snapshots() -> None:
    check_mode_supported(AnchorMode.REPLICATE, SimulatedJudge(supports_pinned_snapshots=True))


def test_other_modes_never_check_for_pinning() -> None:
    judge = SimulatedJudge(supports_pinned_snapshots=False)
    check_mode_supported(AnchorMode.FROZEN_SELF, judge)
    check_mode_supported(AnchorMode.HUMAN, judge)


# --- 3.4 stratified selection -------------------------------------------------------------------


def suite_with_skew(n: int = 1000) -> list[Candidate]:
    """A strongly skewed score distribution across four tags, per the spec's verify."""
    rng = np.random.default_rng(11)
    tags = ["prose", "code", "summary", "qa"]
    return [
        Candidate(
            item_id=f"s{i}",
            score=float(np.clip(rng.beta(5, 1.5), 0, 1)),  # piled up near the ceiling
            tags=(tags[i % 4],),
        )
        for i in range(n)
    ]


def test_stratified_selection_matches_the_suite_marginals() -> None:
    suite = suite_with_skew()
    selection = select_anchors(suite, 260, seed=0)
    assert len(selection) == 260

    coverage = measure_coverage(suite, selection.chosen)
    assert coverage.max_marginal_gap < 0.05, (
        f"stratified selection drifted from the suite by {coverage.max_marginal_gap:.1%}"
    )
    assert not coverage.missing_tags
    assert coverage.decile_coverage == 1.0


def test_selection_is_reproducible_from_the_pinned_seed() -> None:
    suite = suite_with_skew()
    first = select_anchors(suite, 100, seed=42)
    second = select_anchors(suite, 100, seed=42)
    different = select_anchors(suite, 100, seed=43)
    assert first.chosen == second.chosen
    assert first.chosen != different.chosen


def test_selection_is_independent_of_arrival_order() -> None:
    suite = suite_with_skew(300)
    shuffled = list(reversed(suite))
    assert select_anchors(suite, 80, seed=5).chosen == select_anchors(shuffled, 80, seed=5).chosen


def test_stratified_selection_buys_coverage_and_what_it_costs() -> None:
    """The trade-off, made explicit rather than assumed away.

    At a small anchor size, guaranteeing one item per occupied stratum necessarily
    *flattens* the marginals — the anchor set stops looking like the suite in proportion
    in order to look like it in extent. Coverage is the property that protects against a
    region-confined judge change, so it wins; the cost is real and is stated here.
    """
    suite = suite_with_skew()
    stratified = measure_coverage(suite, select_anchors(suite, 40, seed=1).chosen)
    random_pick = measure_coverage(
        suite, select_anchors(suite, 40, seed=1, kind=SelectionKind.RANDOM).chosen
    )
    assert stratified.decile_coverage >= random_pick.decile_coverage
    assert len(stratified.missing_deciles) <= len(random_pick.missing_deciles)
    assert stratified.decile_coverage == 1.0

    # At a size where both can be satisfied, stratified also matches the marginals better.
    big_stratified = measure_coverage(suite, select_anchors(suite, 400, seed=1).chosen)
    big_random = measure_coverage(
        suite, select_anchors(suite, 400, seed=1, kind=SelectionKind.RANDOM).chosen
    )
    assert big_stratified.max_marginal_gap <= big_random.max_marginal_gap


def test_asking_for_more_anchors_than_the_suite_has_is_refused() -> None:
    with pytest.raises(ValueError, match="only 10"):
        select_anchors(suite_with_skew(10), 50)


def test_decile_boundaries() -> None:
    assert decile_of(0.0) == 0
    assert decile_of(0.99) == 9
    assert decile_of(1.0) == 9, "a perfect score belongs in the top decile, not an eleventh"


# --- 3.5 coverage measurement ---------------------------------------------------------------------


def test_a_narrow_anchor_set_produces_a_loud_warning_naming_the_missing_strata() -> None:
    suite = suite_with_skew()
    only_prose = [c.item_id for c in suite if c.tags == ("prose",)][:60]
    coverage = measure_coverage(suite, only_prose)
    assert coverage.missing_tags == ("code", "qa", "summary")
    assert not coverage.adequate
    warnings = coverage.warnings()
    assert any("'code'" in w for w in warnings)
    assert any("invisible" in w for w in warnings)


def test_coverage_is_reported_as_measured_not_guaranteed() -> None:
    suite = suite_with_skew()
    coverage = measure_coverage(suite, select_anchors(suite, 200, seed=0).chosen)
    assert coverage.to_json()["measured_not_guaranteed"] is True
    assert coverage.adequate
    assert coverage.decile_coverage >= COVERAGE_WARN_THRESHOLD


def test_coverage_needs_a_suite_and_overlapping_anchors() -> None:
    with pytest.raises(ValueError, match="empty suite"):
        measure_coverage([], ["a"])
    with pytest.raises(ValueError, match="none of the anchor ids"):
        measure_coverage(suite_with_skew(10), ["nope"])


# --- persistence -------------------------------------------------------------------------------------


def test_anchor_store_round_trips(tmp_path) -> None:
    items = anchor_items(20, gold=True)
    path = tmp_path / "anchors.jsonl"
    save_anchors(items, path)
    loaded = load_anchors(path)
    assert loaded == items


def test_missing_anchor_store_explains_how_to_create_one(tmp_path) -> None:
    with pytest.raises(AnchorModeError, match="benchlock baseline"):
        load_anchors(tmp_path / "nope.jsonl")


def test_the_noise_floor_is_measured_from_the_same_calls_that_freeze_the_baseline() -> None:
    """K replicates do both jobs, so measuring the floor costs nothing extra."""
    judge = SimulatedJudge(seed=9)
    frozen = freeze(anchor_items(100), judge, replicates=5)
    assert judge.calls == 500, "freezing 100 items with K=5 should be exactly 500 calls"
    assert frozen.noise.floor.n_items == 100
    assert frozen.noise.floor.per_item_sd > 0
    assert frozen.noise.floor.run_mean_sd == pytest.approx(
        frozen.noise.floor.per_item_sd / math.sqrt(100), rel=0.8
    )
