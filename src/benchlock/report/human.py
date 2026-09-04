"""The verdict block: the thing a person actually reads.

**The renderer computes nothing.** Every number printed here is a field of `Evidence` or a
property of it; this module formats and lays out, and performs no arithmetic of its own.
That is not fastidiousness. A renderer that did its own arithmetic would be a second,
untested implementation of the decision, free to disagree with `decide()` — and the number
on screen is the one people act on. `tests/test_report.py` enforces the rule by walking
this module's syntax tree and failing on any arithmetic applied to an evidence value.
"""

from __future__ import annotations

import textwrap

from benchlock.model.verdict import Attribution, Provisioning, Verdict

#: Wrap width for the reason list. Wide enough for an interval to stay on one line.
WIDTH = 96

#: What to tell someone once the verdict is in. The judge line is the product: a judge
#: change must not fail your build, it must tell you to re-baseline.
_HEADLINES: dict[Verdict, str] = {
    Verdict.JUDGE: "DO NOT roll back. Re-baseline against the judge's current behaviour:",
    Verdict.SYSTEM: "CI gate: FAIL (exit 1)",
    Verdict.BOTH: "Both moved and benchlock cannot separate them. Investigate before shipping:",
    Verdict.INDETERMINATE: "Fix the provisioning, then this becomes decidable:",
    Verdict.STABLE: "",
}


def _crossing(threshold: float, crossed_at: int | None) -> str:
    if crossed_at is None:
        return f"(threshold {threshold:.1f}, not crossed)"
    return f"(threshold {threshold:.1f}, crossed at run {crossed_at})"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _interval(lower: float, upper: float) -> str:
    return f"[CS: {_signed(lower)}, {_signed(upper)}]"


def _bullet(text: str) -> list[str]:
    """One reason as a wrapped bullet. Explicit newlines in a reason start a new line."""
    out: list[str] = []
    for index, paragraph in enumerate(text.split("\n")):
        prefix = "- " if index == 0 else "  "
        wrapped = textwrap.wrap(paragraph.strip(), width=WIDTH) or [""]
        out.append(f"{prefix}{wrapped[0]}")
        out.extend(f"  {line}" for line in wrapped[1:])
    return out


def render_verdict_block(attribution: Attribution) -> str:
    """The §2 verdict block, exactly."""
    evidence = attribution.evidence
    lines: list[str] = []

    lines.append(
        f"verdict={attribution.verdict.value:<14s} "
        f"confidence: anytime-valid at alpha={attribution.alpha}"
    )
    lines.append(
        f"  E_anchor    = {evidence.e_anchor:>10.1f}   "
        f"{_crossing(evidence.threshold, evidence.crossed_at_anchor)}"
    )
    lines.append(
        f"  E_system    = {evidence.e_system:>10.1f}   "
        f"{_crossing(evidence.threshold, evidence.crossed_at_system)}"
    )
    lines.append(
        f"  E_corrected = {evidence.e_corrected:>10.1f}   "
        f"(threshold {evidence.threshold:.1f}; the system move with the judge's removed)"
    )
    lines.append("")

    # The reasons are produced by the lattice, so the explanation and the decision cannot
    # drift apart.
    for reason in attribution.reasons:
        lines.extend(_bullet(reason))

    if evidence.anchor_n:
        lines.append(f"- anchor provisioning: {evidence.provisioning.value}")
        if evidence.provisioning is Provisioning.ADEQUATE:
            detail = (
                f"minimum detectable judge shift at this anchor size = "
                f"{evidence.min_detectable_judge_shift:.3f}, over {evidence.n_anchor_runs} "
                f"monitored run(s) of {evidence.anchor_n} anchor items"
            )
        else:
            detail = (
                f"at n={evidence.anchor_n} the smallest judge shift detectable by run "
                f"{evidence.n_anchor_runs} is {evidence.min_detectable_judge_shift:.3f}"
            )
        lines.extend(f"  {line}" for line in textwrap.wrap(detail, width=WIDTH))

    if evidence.baseline_judge_model and evidence.judge_pin_changed_at is not None:
        lines.append(f"- baseline judge: {evidence.baseline_judge_model}")
        lines.append(
            f"  current judge:  {evidence.current_judge_model}   "
            f"(declared config changed at run {evidence.judge_pin_changed_at})"
        )
    elif evidence.baseline_judge_model:
        lines.append(f"- judge: {evidence.baseline_judge_model} (config unchanged all epoch)")
    if evidence.judge_pin_delta:
        lines.append(f"- judge pin fields that moved: {', '.join(evidence.judge_pin_delta)}")

    if attribution.epoch_reason:
        lines.append(
            f"- baseline epoch {evidence.epoch}, started because: {attribution.epoch_reason}"
        )

    headline = _HEADLINES[attribution.verdict]
    if headline or attribution.next_command:
        lines.append("")
        if headline:
            lines.append(headline)
        if attribution.next_command:
            lines.append(f"  {attribution.next_command}")

    return "\n".join(lines) + "\n"


def render_shift_summary(attribution: Attribution) -> str:
    """One line for a CI log: the verdict and the two intervals."""
    e = attribution.evidence
    system = _interval(e.system_shift.lower, e.system_shift.upper)
    anchor = _interval(e.anchor_shift.lower, e.anchor_shift.upper)
    corrected = _interval(e.corrected_shift.lower, e.corrected_shift.upper)
    return f"{attribution.verdict.value}: system {system} anchor {anchor} corrected {corrected}"
