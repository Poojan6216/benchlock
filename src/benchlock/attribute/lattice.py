"""The verdict lattice: eight ordered rules, first match wins.

| order | rule_id | condition | verdict |
|---|---|---|---|
| 1 | `pin_violation` | a pin changed without a logged rebaseline | **raise** |
| 2 | `suite_drift` | `suite_hash` differs between runs | **raise** |
| 3 | `insufficient_data` | fewer runs or observations than configured | `STABLE` (low power) |
| 4 | `both_crossed` | anchor crossed and the corrected process crossed | `BOTH` |
| 5 | `judge_only` | anchor crossed and the corrected process did not | `JUDGE` |
| 6 | `race_failed` | system crossed, anchor did not, anchor lacked the power | `INDETERMINATE` |
| 7 | `system_only` | system crossed, anchor did not, anchor had the power | `SYSTEM` |
| 8 | `no_crossing` | neither crossed | `STABLE` |

**Rule 6 before rule 7 is the entire ethical content of the tool, and they must not be
reordered.** Swapping them turns every under-provisioned anchor set into a confident
``SYSTEM`` verdict — a rollback recommendation with nothing behind it. The ordering is
asserted directly in the tests, not merely implied by the code.

The lattice is pure arithmetic on an `Evidence` record. There is no model here, no prompt,
and no heuristic that could be tuned to make a demo look better (Hard Rule 1).
"""

from __future__ import annotations

from collections.abc import Sequence

from benchlock.model.verdict import (
    Attribution,
    AttributionRefusedError,
    Evidence,
    Provisioning,
    Verdict,
)

#: The rules in evaluation order. Exposed so tests can assert the order itself.
RULE_ORDER: tuple[str, ...] = (
    "pin_violation",
    "suite_drift",
    "insufficient_data",
    "both_crossed",
    "judge_only",
    "race_failed",
    "system_only",
    "no_crossing",
)


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


def _interval(evidence_interval: object) -> str:
    from benchlock.stats.confseq import Interval

    assert isinstance(evidence_interval, Interval)
    return f"[CS: {_fmt(evidence_interval.lower)}, {_fmt(evidence_interval.upper)}]"


def apply_lattice(
    evidence: Evidence,
    alpha: float,
    *,
    target_shift: float = 0.05,
    min_runs: int,
    min_obs: int,
    observations_in_last_run: int,
    pin_delta: Sequence[str] = (),
    pin_rebaselined: bool = True,
    suite_hashes_agree: bool = True,
    epoch_reason: str = "",
) -> Attribution:
    """Evaluate the eight rules in order and return the first match.

    Rules 1 and 2 raise `AttributionRefusedError` rather than returning a verdict: they
    signal that the two numbers being compared are not measurements of the same thing.
    """
    # --- 1. pin_violation (Hard Rule 8) ---------------------------------------------------
    if pin_delta and not pin_rebaselined:
        raise AttributionRefusedError(
            "pin_violation",
            "cannot attribute: the judge or anchor pin changed without a logged "
            f"rebaseline ({', '.join(pin_delta)})",
            "scores before and after the change are not comparable. Record the change "
            "with `benchlock rebaseline --reason judge-version-change`",
        )

    # --- 2. suite_drift (Hard Rule 6) -----------------------------------------------------
    if not suite_hashes_agree:
        raise AttributionRefusedError(
            "suite_drift",
            "cannot attribute: the eval suite's item set changed between runs",
            "attribution separates the judge from the system *given a fixed suite*. If "
            "your eval inputs move, score movement has a third cause benchlock does not "
            "model. Fix the suite, or stop using attribution mode on this stream",
        )

    # --- 3. insufficient_data -------------------------------------------------------------
    if evidence.n_runs < min_runs or observations_in_last_run < min_obs:
        return Attribution(
            verdict=Verdict.STABLE,
            rule_id="insufficient_data",
            reasons=(
                f"low_power: {evidence.n_runs} run(s) of a required {min_runs}, "
                f"{observations_in_last_run} observation(s) of a required {min_obs}",
                "no verdict is attempted this early; a stream too short to be informative "
                "reads as stable rather than as evidence of anything",
            ),
            evidence=evidence,
            alpha=alpha,
            target_shift=target_shift,
            epoch_reason=epoch_reason,
        )

    # "the anchor-corrected system interval excludes 0" in the spec's table. We test the
    # corrected *process* rather than the interval: the difference-in-differences stream is
    # monitored directly, which is an anytime-valid test rather than a comparison of two
    # intervals whose widths add (Hard Rule 3).
    corrected_excludes_zero = evidence.corrected_crossed

    # --- 4. both_crossed ------------------------------------------------------------------
    # The spec's table conditions this on `E_system >= thr` as well. We do not, and the
    # reason is **cancellation**: a judge that got stricter while the system improved by
    # the same amount leaves the raw system score perfectly flat. `E_system` never crosses,
    # both components really did move, and requiring the system stream to have crossed
    # would return STABLE for a pipeline where two real changes happened to offset. The
    # corrected process sees it, because the difference of the two deviations is exactly
    # what did not cancel. Phase 7.3 measures this case; catching it is the point.
    if evidence.anchor_crossed and corrected_excludes_zero:
        return Attribution(
            verdict=Verdict.BOTH,
            rule_id="both_crossed",
            reasons=(
                f"the anchor set moved {_fmt(evidence.anchor_point_shift)} "
                f"{_interval(evidence.anchor_shift)}",
                f"your system moved {_fmt(evidence.system_point_shift)} raw; "
                f"anchor-corrected {_fmt(evidence.corrected_shift.midpoint)} "
                f"{_interval(evidence.corrected_shift)}",
                "both moved, and removing the judge's movement does not explain away the "
                "system's — they are confounded and benchlock will not pick one",
            ),
            evidence=evidence,
            alpha=alpha,
            target_shift=target_shift,
            next_command="benchlock report",
            epoch_reason=epoch_reason,
        )

    # --- 5. judge_only --------------------------------------------------------------------
    if evidence.anchor_crossed and not corrected_excludes_zero:
        reasons = [
            f"the anchor set moved {_fmt(evidence.anchor_point_shift)} "
            f"{_interval(evidence.anchor_shift)}",
            "the system under test does not touch the anchor set; only the judge can move it",
            f"your system moved {_fmt(evidence.system_point_shift)} raw; anchor-corrected "
            f"{_fmt(evidence.corrected_shift.midpoint)} {_interval(evidence.corrected_shift)}",
        ]
        if evidence.judge_pin_delta:
            reasons.append(f"declared judge config changed: {', '.join(evidence.judge_pin_delta)}")
        return Attribution(
            verdict=Verdict.JUDGE,
            rule_id="judge_only",
            reasons=tuple(reasons),
            evidence=evidence,
            alpha=alpha,
            target_shift=target_shift,
            next_command="benchlock rebaseline --reason judge-version-change",
            epoch_reason=epoch_reason,
        )

    # --- 6. race_failed (Hard Rule 2) — MUST precede rule 7 --------------------------------
    if (
        evidence.system_crossed
        and not evidence.anchor_crossed
        and evidence.provisioning is Provisioning.UNDER_PROVISIONED
    ):
        return Attribution(
            verdict=Verdict.INDETERMINATE,
            rule_id="race_failed",
            reasons=(
                "your system score moved, and the anchor set did not",
                f"BUT the anchor set is under-provisioned: at n={evidence.anchor_n} the "
                f"minimum judge shift this anchor process could have detected by run "
                f"{evidence.n_anchor_runs} is "
                f"{evidence.min_detectable_judge_shift:.3f}. The observed system move is "
                f"{abs(evidence.system_point_shift):.3f}. A judge shift of that size "
                "would be INVISIBLE here",
                "benchlock will not attribute this to your system. It cannot rule out the judge",
            ),
            evidence=evidence,
            alpha=alpha,
            target_shift=target_shift,
            next_command="benchlock plan --target-shift 0.05",
            epoch_reason=epoch_reason,
        )

    # --- 7. system_only -------------------------------------------------------------------
    if evidence.system_crossed and not evidence.anchor_crossed:
        return Attribution(
            verdict=Verdict.SYSTEM,
            rule_id="system_only",
            reasons=(
                f"the anchor set is stable: {_fmt(evidence.anchor_point_shift)} "
                f"{_interval(evidence.anchor_shift)}",
                f"your system moved {_fmt(evidence.system_point_shift)} "
                f"{_interval(evidence.system_shift)}",
                f"the anchor set could have detected a judge shift as small as "
                f"{evidence.min_detectable_judge_shift:.3f}, which is smaller than the "
                f"observed move, so the judge is ruled out",
            ),
            evidence=evidence,
            alpha=alpha,
            target_shift=target_shift,
            next_command="benchlock report",
            epoch_reason=epoch_reason,
        )

    # --- 8. no_crossing -------------------------------------------------------------------
    return Attribution(
        verdict=Verdict.STABLE,
        rule_id="no_crossing",
        reasons=(
            f"neither process has crossed the threshold of {evidence.threshold:.1f}",
            f"system e-value {evidence.e_system:.1f}, anchor e-value {evidence.e_anchor:.1f}",
        ),
        evidence=evidence,
        alpha=alpha,
        target_shift=target_shift,
        epoch_reason=epoch_reason,
    )
