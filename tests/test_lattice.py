"""Phase 2.1 verify: the eight rules, every branch, and both boundaries.

The ordering assertions here are not decoration. Rule 6 (`race_failed`) sits before rule 7
(`system_only`) because an anchor set too small to have seen a judge shift must not produce
a confident `SYSTEM` verdict. Swapping them would make every under-provisioned setup
recommend a rollback with nothing behind it, so the order is asserted directly.
"""

from __future__ import annotations

import pytest

from benchlock.attribute.lattice import RULE_ORDER, apply_lattice
from benchlock.model.verdict import (
    AttributionRefusedError,
    Evidence,
    Provisioning,
    Verdict,
)
from benchlock.stats.confseq import Interval

THRESHOLD = 40.0  # 1 / alpha_monitor at alpha=0.05


def evidence(
    *,
    e_system: float = 1.0,
    e_anchor: float = 1.0,
    e_corrected: float = 1.0,
    system_shift: tuple[float, float] = (-0.09, -0.05),
    anchor_shift: tuple[float, float] = (-0.01, 0.01),
    corrected_shift: tuple[float, float] = (-0.09, -0.05),
    mds: float = 0.02,
    provisioning: Provisioning = Provisioning.ADEQUATE,
    n_runs: int = 40,
    n_anchor_runs: int = 40,
    anchor_n: int = 200,
    judge_pin_delta: tuple[str, ...] = (),
) -> Evidence:
    return Evidence(
        e_system=e_system,
        e_anchor=e_anchor,
        e_corrected=e_corrected,
        threshold=THRESHOLD,
        system_shift=Interval(*system_shift),
        anchor_shift=Interval(*anchor_shift),
        corrected_shift=Interval(*corrected_shift),
        crossed_at_system=10 if e_system >= THRESHOLD else None,
        crossed_at_anchor=10 if e_anchor >= THRESHOLD else None,
        min_detectable_judge_shift=mds,
        provisioning=provisioning,
        judge_pin_delta=judge_pin_delta,
        n_runs=n_runs,
        n_anchor_runs=n_anchor_runs,
        anchor_n=anchor_n,
    )


def decide_with(ev: Evidence, **kwargs) -> object:
    params = {"min_runs": 8, "min_obs": 30, "observations_in_last_run": 200}
    params.update(kwargs)
    return apply_lattice(ev, 0.05, **params)


# --- the order itself ------------------------------------------------------------------


def test_the_rule_order_is_the_documented_one() -> None:
    assert RULE_ORDER == (
        "pin_violation",
        "suite_drift",
        "insufficient_data",
        "both_crossed",
        "judge_only",
        "race_failed",
        "system_only",
        "no_crossing",
    )


def test_race_failed_precedes_system_only() -> None:
    """Hard Rule 2, asserted as an ordering fact rather than left implicit in the code."""
    assert RULE_ORDER.index("race_failed") < RULE_ORDER.index("system_only")


def test_the_raising_rules_come_first() -> None:
    assert RULE_ORDER.index("pin_violation") == 0
    assert RULE_ORDER.index("suite_drift") == 1


# --- rule 1: pin_violation --------------------------------------------------------------


def test_rule1_pin_violation_raises() -> None:
    with pytest.raises(AttributionRefusedError) as excinfo:
        decide_with(evidence(), pin_delta=("model",), pin_rebaselined=False)
    assert excinfo.value.rule_id == "pin_violation"
    assert "model" in excinfo.value.message
    assert "benchlock rebaseline" in excinfo.value.hint


def test_rule1_a_declared_rebaseline_is_not_a_violation() -> None:
    result = decide_with(evidence(), pin_delta=("model",), pin_rebaselined=True)
    assert result.verdict is Verdict.STABLE


def test_rule1_beats_every_other_rule() -> None:
    # Even with a screaming system signal, an undeclared pin change refuses first.
    with pytest.raises(AttributionRefusedError):
        decide_with(
            evidence(e_system=1e9, e_corrected=1e9),
            pin_delta=("rubric_hash",),
            pin_rebaselined=False,
        )


# --- rule 2: suite_drift ------------------------------------------------------------------


def test_rule2_suite_drift_raises() -> None:
    with pytest.raises(AttributionRefusedError) as excinfo:
        decide_with(evidence(), suite_hashes_agree=False)
    assert excinfo.value.rule_id == "suite_drift"
    assert "third cause" in excinfo.value.hint


def test_rule2_beats_everything_after_it() -> None:
    with pytest.raises(AttributionRefusedError) as excinfo:
        decide_with(evidence(e_system=1e9, e_anchor=1e9), suite_hashes_agree=False)
    assert excinfo.value.rule_id == "suite_drift"


# --- rule 3: insufficient_data --------------------------------------------------------------


@pytest.mark.parametrize("n_runs", [0, 1, 5, 7])
def test_rule3_too_few_runs_is_stable_with_low_power(n_runs: int) -> None:
    result = decide_with(evidence(n_runs=n_runs, e_system=1e9, e_corrected=1e9))
    assert result.verdict is Verdict.STABLE
    assert result.rule_id == "insufficient_data"
    assert any("low_power" in r for r in result.reasons)


def test_rule3_too_few_observations_is_stable() -> None:
    result = decide_with(evidence(e_system=1e9), observations_in_last_run=10)
    assert result.rule_id == "insufficient_data"


def test_rule3_boundary_exactly_at_min_runs_proceeds() -> None:
    assert decide_with(evidence(n_runs=8)).rule_id != "insufficient_data"
    assert decide_with(evidence(n_runs=7)).rule_id == "insufficient_data"


def test_rule3_boundary_exactly_at_min_obs_proceeds() -> None:
    assert decide_with(evidence(), observations_in_last_run=30).rule_id != "insufficient_data"
    assert decide_with(evidence(), observations_in_last_run=29).rule_id == "insufficient_data"


# --- rule 4: both_crossed --------------------------------------------------------------------


def test_rule4_both_crossed_with_a_surviving_correction() -> None:
    result = decide_with(evidence(e_system=1e5, e_anchor=1e5, e_corrected=1e5))
    assert result.verdict is Verdict.BOTH
    assert result.rule_id == "both_crossed"
    assert any("confounded" in r for r in result.reasons)


def test_rule4_needs_the_anchor_and_the_correction_to_have_crossed() -> None:
    # Anchor and system crossed but the correction did not: the judge explains it all.
    result = decide_with(evidence(e_system=1e5, e_anchor=1e5, e_corrected=1.0))
    assert result.verdict is Verdict.JUDGE


def test_rule4_catches_cancellation_where_the_system_stream_shows_nothing() -> None:
    """The nastiest case in the design: two real moves that offset.

    The judge got stricter and the system improved by the same amount, so the raw system
    score is perfectly flat and `E_system` never crosses. Both components moved. A
    single-stream detector sees a healthy pipeline; the corrected process does not.
    """
    result = decide_with(evidence(e_system=1.0, e_anchor=1e5, e_corrected=1e5))
    assert result.verdict is Verdict.BOTH
    assert result.rule_id == "both_crossed"


def test_rule4_boundary_exactly_at_threshold_counts_as_crossed() -> None:
    result = decide_with(evidence(e_system=THRESHOLD, e_anchor=THRESHOLD, e_corrected=THRESHOLD))
    assert result.verdict is Verdict.BOTH
    just_under = decide_with(
        evidence(e_system=THRESHOLD, e_anchor=THRESHOLD, e_corrected=THRESHOLD - 1e-9)
    )
    assert just_under.verdict is Verdict.JUDGE


# --- rule 5: judge_only ------------------------------------------------------------------------


def test_rule5_judge_only_when_the_anchor_moved_and_the_correction_did_not() -> None:
    result = decide_with(
        evidence(e_system=1.0, e_anchor=1e5, e_corrected=1.0, anchor_shift=(-0.14, -0.09))
    )
    assert result.verdict is Verdict.JUDGE
    assert result.rule_id == "judge_only"
    assert any("only the judge can" in r for r in result.reasons)
    assert result.next_command.startswith("benchlock rebaseline")


def test_rule5_fires_even_when_the_system_also_crossed() -> None:
    """A judge shift moves both streams; that is the phantom regression."""
    result = decide_with(evidence(e_system=1e5, e_anchor=1e5, e_corrected=1.0))
    assert result.verdict is Verdict.JUDGE


def test_rule5_reports_a_declared_pin_change() -> None:
    result = decide_with(
        evidence(e_anchor=1e5, e_corrected=1.0, judge_pin_delta=("model",)),
        pin_delta=("model",),
        pin_rebaselined=True,
    )
    assert any("declared judge config changed" in r for r in result.reasons)


def test_rule5_boundary_anchor_exactly_at_threshold() -> None:
    assert decide_with(evidence(e_anchor=THRESHOLD, e_corrected=1.0)).verdict is Verdict.JUDGE
    assert (
        decide_with(evidence(e_anchor=THRESHOLD - 1e-9, e_corrected=1.0)).verdict is Verdict.STABLE
    )


# --- rule 6: race_failed (Hard Rule 2) -----------------------------------------------------------


def test_rule6_under_provisioned_anchor_gives_indeterminate() -> None:
    result = decide_with(
        evidence(
            e_system=1e5,
            e_anchor=1.0,
            e_corrected=1e5,
            mds=0.094,
            provisioning=Provisioning.UNDER_PROVISIONED,
            anchor_n=40,
        )
    )
    assert result.verdict is Verdict.INDETERMINATE
    assert result.rule_id == "race_failed"
    assert any("INVISIBLE" in r for r in result.reasons)
    assert any("will not attribute" in r for r in result.reasons)
    assert result.next_command.startswith("benchlock plan")


def test_rule6_beats_rule7_on_identical_evidence() -> None:
    """The one ordering that matters most: identical inputs, only provisioning differs."""
    base = {"e_system": 1e5, "e_anchor": 1.0, "e_corrected": 1e5}
    under = decide_with(evidence(**base, provisioning=Provisioning.UNDER_PROVISIONED))
    adequate = decide_with(evidence(**base, provisioning=Provisioning.ADEQUATE))
    assert under.verdict is Verdict.INDETERMINATE
    assert adequate.verdict is Verdict.SYSTEM


def test_rule6_never_fires_when_the_anchor_itself_crossed() -> None:
    # If the anchor moved, we are in judge/both territory and provisioning is moot.
    result = decide_with(
        evidence(
            e_system=1e5,
            e_anchor=1e5,
            e_corrected=1e5,
            provisioning=Provisioning.UNDER_PROVISIONED,
        )
    )
    assert result.verdict is Verdict.BOTH


# --- rule 7: system_only ----------------------------------------------------------------------------


def test_rule7_system_only_when_the_anchor_had_the_power() -> None:
    result = decide_with(evidence(e_system=1e5, e_anchor=1.0, e_corrected=1e5, mds=0.02))
    assert result.verdict is Verdict.SYSTEM
    assert result.rule_id == "system_only"
    assert any("anchor set is stable" in r for r in result.reasons)
    assert any("judge is ruled out" in r for r in result.reasons)


def test_rule7_boundary_system_exactly_at_threshold() -> None:
    assert decide_with(evidence(e_system=THRESHOLD, e_corrected=1e5)).verdict is Verdict.SYSTEM
    assert (
        decide_with(evidence(e_system=THRESHOLD - 1e-9, e_corrected=1e5)).verdict is Verdict.STABLE
    )


# --- rule 8: no_crossing ------------------------------------------------------------------------------


def test_rule8_nothing_crossed_is_stable() -> None:
    result = decide_with(evidence(e_system=1.0, e_anchor=1.0, e_corrected=1.0))
    assert result.verdict is Verdict.STABLE
    assert result.rule_id == "no_crossing"
    assert any("neither process" in r for r in result.reasons)


def test_rule8_a_crossed_correction_alone_does_not_decide() -> None:
    # The corrected process is a modifier on the other two, never a verdict on its own.
    result = decide_with(evidence(e_system=1.0, e_anchor=1.0, e_corrected=1e9))
    assert result.verdict is Verdict.STABLE


# --- properties of every branch --------------------------------------------------------------------------


ALL_BRANCHES = [
    ("insufficient_data", {"n_runs": 2}, {}),
    ("both_crossed", {"e_system": 1e5, "e_anchor": 1e5, "e_corrected": 1e5}, {}),
    ("judge_only", {"e_anchor": 1e5, "e_corrected": 1.0}, {}),
    (
        "race_failed",
        {
            "e_system": 1e5,
            "e_corrected": 1e5,
            "provisioning": Provisioning.UNDER_PROVISIONED,
        },
        {},
    ),
    ("system_only", {"e_system": 1e5, "e_corrected": 1e5}, {}),
    ("no_crossing", {}, {}),
]


@pytest.mark.parametrize(
    ("rule_id", "kwargs", "extra"), ALL_BRANCHES, ids=[b[0] for b in ALL_BRANCHES]
)
def test_every_branch_is_reachable_and_names_itself(
    rule_id: str, kwargs: dict, extra: dict
) -> None:
    result = decide_with(evidence(**kwargs), **extra)
    assert result.rule_id == rule_id
    assert result.rule_id, "rule_id must never be empty"
    assert result.reasons, "every verdict must explain itself"
    assert result.evidence is not None
    assert result.alpha == 0.05


def test_all_six_returning_rules_are_covered_by_the_branch_table() -> None:
    covered = {b[0] for b in ALL_BRANCHES}
    returning = set(RULE_ORDER) - {"pin_violation", "suite_drift"}
    assert covered == returning


def test_an_attribution_must_name_a_rule() -> None:
    from benchlock.model.verdict import Attribution

    with pytest.raises(ValueError, match="must name the lattice rule"):
        Attribution(verdict=Verdict.STABLE, rule_id="", reasons=(), evidence=evidence(), alpha=0.05)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"e_system": 1e5, "e_anchor": 1e5, "e_corrected": 1e5}, Verdict.BOTH),
        ({"e_anchor": 1e5, "e_corrected": 1.0}, Verdict.JUDGE),
        ({"e_system": 1e5, "e_anchor": 1e5, "e_corrected": 1.0}, Verdict.JUDGE),
        (
            {
                "e_system": 1e5,
                "e_corrected": 1e5,
                "provisioning": Provisioning.UNDER_PROVISIONED,
            },
            Verdict.INDETERMINATE,
        ),
        ({"e_system": 1e5, "e_corrected": 1e5}, Verdict.SYSTEM),
        ({}, Verdict.STABLE),
        ({"e_system": 1e5, "e_corrected": 1.0}, Verdict.SYSTEM),
        # cancellation: the anchor moved, the raw system score did not, the difference did
        ({"e_anchor": 1e5, "e_system": 1.0, "e_corrected": 1e5}, Verdict.BOTH),
    ],
)
def test_verdict_truth_table(kwargs: dict, expected: Verdict) -> None:
    assert decide_with(evidence(**kwargs)).verdict is expected
