"""The five verdicts, and the evidence that produces them.

The verdict set is deliberately five and not three. ``INDETERMINATE`` is the one that
makes the other four mean anything: without it, a tool whose anchor set was too small to
rule out the judge would still say ``SYSTEM``, and be right by luck often enough to be
believed. A tool that is confidently wrong one time in ten is worse than useless in a CI
gate, because people learn to ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from benchlock.model.streams import StreamKind
from benchlock.stats.confseq import Interval


class Verdict(StrEnum):
    STABLE = "stable"  # neither process has crossed
    JUDGE = "judge"  # the judge moved
    SYSTEM = "system"  # the system moved, and the judge demonstrably did not
    BOTH = "both"  # both moved; anchor correction does not explain the system move
    INDETERMINATE = "indeterminate"  # cannot rule out the judge. Hard Rule 2.


class Provisioning(StrEnum):
    ADEQUATE = "ADEQUATE"
    UNDER_PROVISIONED = "UNDER_PROVISIONED"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Everything the verdict was derived from. The renderer computes nothing itself."""

    e_system: float  # e-detector value for the system stream
    e_anchor: float  # e-detector value for the anchor stream
    #: e-detector value for the difference-in-differences stream: the system's movement
    #: with the judge's removed run by run. This is what decides whether a system move
    #: survives the anchor correction.
    e_corrected: float
    threshold: float  # 1/alpha
    system_shift: Interval  # confidence sequence, anytime-valid
    anchor_shift: Interval
    corrected_shift: Interval  # system shift with the anchor shift removed
    #: Stored as two fields rather than a dict so `Evidence` stays immutable and hashable;
    #: `crossed_at` below reassembles the mapping the spec describes.
    crossed_at_system: int | None
    crossed_at_anchor: int | None
    min_detectable_judge_shift: float  # what the anchor process could have caught by now
    provisioning: Provisioning
    judge_pin_delta: tuple[str, ...] = ()  # which judge fields changed, if declared
    n_runs: int = 0
    n_anchor_runs: int = 0
    anchor_n: int = 0
    epoch: int = 0
    baseline_judge_model: str = ""
    current_judge_model: str = ""
    judge_pin_changed_at: int | None = None

    @property
    def crossed_at(self) -> dict[StreamKind, int | None]:
        return {
            StreamKind.SYSTEM: self.crossed_at_system,
            StreamKind.ANCHOR: self.crossed_at_anchor,
        }

    @property
    def system_crossed(self) -> bool:
        return self.e_system >= self.threshold

    @property
    def anchor_crossed(self) -> bool:
        return self.e_anchor >= self.threshold

    @property
    def corrected_crossed(self) -> bool:
        """Did the system move survive removing the judge's movement?"""
        return self.e_corrected >= self.threshold

    @property
    def system_point_shift(self) -> float:
        """Midpoint of the system shift interval. For display and the race check."""
        return self.system_shift.midpoint

    @property
    def anchor_point_shift(self) -> float:
        return self.anchor_shift.midpoint

    def to_json(self) -> dict[str, Any]:
        return {
            "e_system": self.e_system,
            "e_anchor": self.e_anchor,
            "e_corrected": self.e_corrected,
            "threshold": self.threshold,
            "system_shift": self.system_shift.to_json(),
            "anchor_shift": self.anchor_shift.to_json(),
            "corrected_shift": self.corrected_shift.to_json(),
            "crossed_at": {"system": self.crossed_at_system, "anchor": self.crossed_at_anchor},
            "min_detectable_judge_shift": self.min_detectable_judge_shift,
            "provisioning": self.provisioning.value,
            "judge_pin_delta": list(self.judge_pin_delta),
            "n_runs": self.n_runs,
            "n_anchor_runs": self.n_anchor_runs,
            "anchor_n": self.anchor_n,
            "epoch": self.epoch,
            "baseline_judge_model": self.baseline_judge_model,
            "current_judge_model": self.current_judge_model,
            "judge_pin_changed_at": self.judge_pin_changed_at,
        }


@dataclass(frozen=True, slots=True)
class Attribution:
    """The output of `decide()`. `rule_id` names the lattice branch that fired."""

    verdict: Verdict
    rule_id: str  # which lattice rule fired. never empty.
    reasons: tuple[str, ...]  # human-readable, shown in the verdict block
    evidence: Evidence
    alpha: float
    next_command: str = ""  # the command that acts on this verdict, if any
    epoch_reason: str = ""  # why the current baseline epoch was started

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("every attribution must name the lattice rule that produced it")

    def to_json(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "rule_id": self.rule_id,
            "reasons": list(self.reasons),
            "evidence": self.evidence.to_json(),
            "alpha": self.alpha,
            "next_command": self.next_command,
            "epoch_reason": self.epoch_reason,
        }


class AttributionRefusedError(Exception):
    """Raised by lattice rules 1 and 2, which are errors rather than verdicts.

    A pin that moved without a rebaseline, or an eval suite that changed shape, means the
    two numbers being compared are not measurements of the same thing. Returning a verdict
    would be reporting the difference between two different experiments as a result.
    """

    def __init__(self, rule_id: str, message: str, hint: str = "") -> None:
        self.rule_id = rule_id
        self.message = message
        self.hint = hint
        super().__init__(f"{message}" + (f"\n  fix: {hint}" if hint else ""))


@dataclass(frozen=True, slots=True)
class Rule:
    """One row of the lattice. Evaluated in order; first match wins."""

    order: int
    rule_id: str
    verdict: Verdict | None  # None for the two raising rules
    description: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
