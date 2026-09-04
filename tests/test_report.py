"""Phase 2.5 verify: the verdict block, and the rule that the renderer computes nothing."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from benchlock.model.verdict import (
    Attribution,
    Evidence,
    Provisioning,
    Verdict,
)
from benchlock.report import human
from benchlock.report.human import render_shift_summary, render_verdict_block
from benchlock.stats.confseq import Interval


def attribution_for(verdict: Verdict) -> Attribution:
    evidence = Evidence(
        e_system=148208.9,
        e_anchor=0.5,
        e_corrected=554013.8,
        threshold=40.0,
        system_shift=Interval(-0.107, 0.031),
        anchor_shift=Interval(-0.063, 0.065),
        corrected_shift=Interval(-0.115, -0.003),
        crossed_at_system=27,
        crossed_at_anchor=None,
        min_detectable_judge_shift=0.032,
        provisioning=Provisioning.ADEQUATE,
        judge_pin_delta=(),
        n_runs=60,
        n_anchor_runs=55,
        anchor_n=200,
        baseline_judge_model="claude-sonnet-4-5-20250929",
        current_judge_model="claude-sonnet-4-5-20250929",
    )
    return Attribution(
        verdict=verdict,
        rule_id="system_only",
        reasons=("the anchor set is stable", "your system moved -0.038"),
        evidence=evidence,
        alpha=0.05,
        next_command="benchlock report",
    )


# --- the rule: no arithmetic in the renderer -----------------------------------------------


@pytest.mark.mandatory
def test_the_renderer_performs_no_arithmetic() -> None:
    """Every number shown comes from `Evidence`; this module formats and nothing else.

    A renderer doing its own arithmetic would be a second, untested implementation of the
    decision, free to disagree with `decide()` — and the number on screen is the one
    people act on. Enforced by walking the syntax tree rather than by convention.
    """
    source = Path(inspect.getfile(human)).read_text()
    tree = ast.parse(source)

    # `int | None` is a BinOp too, so match on the arithmetic operators specifically
    # rather than on "any binary operation".
    arithmetic = (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.MatMult)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, arithmetic):
            offenders.append(f"line {node.lineno}: {type(node.op).__name__}")
        # String concatenation and % formatting are layout. Numeric addition is not.
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant) and isinstance(side.value, (int, float)):
                    offenders.append(f"line {node.lineno}: numeric {type(node.op).__name__}")
    assert not offenders, "the renderer must not compute:\n" + "\n".join(offenders)


@pytest.mark.mandatory
def test_the_renderer_reads_evidence_without_transforming_it() -> None:
    """No `evidence.<field>` may appear inside an arithmetic expression."""
    source = Path(inspect.getfile(human)).read_text()
    tree = ast.parse(source)

    def mentions_evidence(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id in {"evidence", "e"}
            for child in ast.walk(node)
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and mentions_evidence(node):
            raise AssertionError(
                f"line {node.lineno}: arithmetic applied to an evidence field. "
                "The renderer must format, not compute"
            )


# --- golden output over all five verdicts ------------------------------------------------------


@pytest.mark.parametrize("verdict", list(Verdict))
def test_every_verdict_renders(verdict: Verdict) -> None:
    block = render_verdict_block(attribution_for(verdict))
    assert f"verdict={verdict.value}" in block
    assert "anytime-valid at alpha=0.05" in block
    assert "E_anchor" in block and "E_system" in block and "E_corrected" in block
    assert "anchor provisioning:" in block
    assert block.endswith("\n")


def test_the_block_shows_the_threshold_and_crossing_run() -> None:
    block = render_verdict_block(attribution_for(Verdict.SYSTEM))
    assert "(threshold 40.0, crossed at run 27)" in block
    assert "(threshold 40.0, not crossed)" in block


def test_judge_verdict_tells_you_not_to_roll_back() -> None:
    attribution = attribution_for(Verdict.JUDGE)
    block = render_verdict_block(attribution)
    assert "DO NOT roll back" in block
    assert "benchlock report" in block  # the next_command from the fixture


def test_system_verdict_says_the_gate_fails() -> None:
    assert "CI gate: FAIL (exit 1)" in render_verdict_block(attribution_for(Verdict.SYSTEM))


def test_indeterminate_verdict_says_what_to_fix() -> None:
    block = render_verdict_block(attribution_for(Verdict.INDETERMINATE))
    assert "Fix the provisioning" in block


def test_under_provisioned_block_names_the_anchor_size_and_run() -> None:
    base = attribution_for(Verdict.INDETERMINATE)
    evidence = base.evidence
    under = Attribution(
        verdict=Verdict.INDETERMINATE,
        rule_id="race_failed",
        reasons=("your system score moved, and the anchor set did not",),
        evidence=Evidence(
            **{
                **{
                    f.name: getattr(evidence, f.name)
                    for f in evidence.__dataclass_fields__.values()
                },
                "provisioning": Provisioning.UNDER_PROVISIONED,
                "anchor_n": 40,
                "min_detectable_judge_shift": 0.094,
            }
        ),
        alpha=0.05,
        next_command="benchlock plan --target-shift 0.05",
    )
    block = render_verdict_block(under)
    assert "UNDER_PROVISIONED" in block
    assert "at n=40" in block
    assert "0.094" in block
    assert "benchlock plan --target-shift 0.05" in block


def test_a_declared_pin_change_is_shown() -> None:
    base = attribution_for(Verdict.JUDGE)
    evidence = base.evidence
    changed = Attribution(
        verdict=Verdict.JUDGE,
        rule_id="judge_only",
        reasons=("the anchor set moved",),
        evidence=Evidence(
            **{
                **{
                    f.name: getattr(evidence, f.name)
                    for f in evidence.__dataclass_fields__.values()
                },
                "current_judge_model": "claude-sonnet-4-5-20260114",
                "judge_pin_changed_at": 116,
                "judge_pin_delta": ("model",),
            }
        ),
        alpha=0.05,
        next_command="benchlock rebaseline --reason judge-version-change",
    )
    block = render_verdict_block(changed)
    assert "baseline judge: claude-sonnet-4-5-20250929" in block
    assert "current judge:  claude-sonnet-4-5-20260114" in block
    assert "declared config changed at run 116" in block
    assert "judge pin fields that moved: model" in block


def test_no_line_is_absurdly_long() -> None:
    for verdict in Verdict:
        for line in render_verdict_block(attribution_for(verdict)).splitlines():
            assert len(line) <= 110, f"line too long for a terminal: {line!r}"


def test_shift_summary_is_one_line() -> None:
    summary = render_shift_summary(attribution_for(Verdict.SYSTEM))
    assert "\n" not in summary
    assert summary.startswith("system:")
    assert "corrected [CS:" in summary


def test_the_block_is_stable_for_the_same_attribution() -> None:
    attribution = attribution_for(Verdict.SYSTEM)
    assert render_verdict_block(attribution) == render_verdict_block(attribution)
