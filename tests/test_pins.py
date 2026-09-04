"""Phase 0.5 verify: the eight pin cases, each producing the right delta and message."""

from __future__ import annotations

from dataclasses import replace

import pytest

from benchlock.model.pins import (
    AnchorMode,
    AnchorPin,
    JudgePin,
    NoiseFloor,
    PinViolationError,
    baseline_scores_hash,
    check_anchor_pin,
    check_judge_pin,
    sha256_ids,
)

RUBRIC = "Score the answer 1-5 for helpfulness.\nBe strict about factual errors.\n"
PARAMS = {"temperature": 0.0, "max_tokens": 512}


def make_judge_pin(**over: object) -> JudgePin:
    kwargs: dict[str, object] = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "rubric_text": RUBRIC,
        "params": dict(PARAMS),
        "scale": (1.0, 5.0),
    }
    kwargs.update(over)
    return JudgePin.build(**kwargs)  # type: ignore[arg-type]


def make_anchor_pin(scores: dict[str, float] | None = None) -> AnchorPin:
    scores = scores if scores is not None else {f"a{i}": 0.6 + i * 0.001 for i in range(50)}
    return AnchorPin(
        mode=AnchorMode.FROZEN_SELF,
        item_set_hash=sha256_ids(scores),
        baseline_scores_hash=baseline_scores_hash(scores),
        n=len(scores),
        noise_floor=NoiseFloor(per_item_sd=0.08, run_mean_sd=0.01, replicates=5, n_items=50),
    )


# --- the eight cases ---------------------------------------------------------------------


def test_case_1_nothing_changed_is_silent() -> None:
    pin = make_judge_pin()
    check_judge_pin(pin, make_judge_pin())  # must not raise
    assert pin.differs_from(make_judge_pin()) == ()
    anchor = make_anchor_pin()
    check_anchor_pin(anchor, make_anchor_pin())


def test_case_2_model_string_changed() -> None:
    with pytest.raises(PinViolationError) as excinfo:
        check_judge_pin(make_judge_pin(model="claude-sonnet-4-5-20260114"), make_judge_pin())
    exc = excinfo.value
    assert exc.changed == ("model",)
    assert "judge model snapshot changed" in exc.message
    assert "claude-sonnet-4-5-20250929" in exc.message
    assert "benchlock rebaseline --reason judge-version-change" in exc.hint


def test_case_3_rubric_whitespace_only_change_still_trips() -> None:
    # Whitespace changes prompts. A pin that ignored it would be a pin that lies.
    whitespaced = RUBRIC.replace("1-5", "1-5 ")
    assert whitespaced != RUBRIC
    assert whitespaced.split() == RUBRIC.split(), "this case must differ ONLY in whitespace"
    with pytest.raises(PinViolationError) as excinfo:
        check_judge_pin(make_judge_pin(rubric_text=whitespaced), make_judge_pin())
    assert excinfo.value.changed == ("rubric_hash",)
    assert "including whitespace" in excinfo.value.message


def test_case_4_temperature_changed() -> None:
    with pytest.raises(PinViolationError) as excinfo:
        check_judge_pin(
            make_judge_pin(params={"temperature": 0.7, "max_tokens": 512}), make_judge_pin()
        )
    assert excinfo.value.changed == ("params_hash",)
    assert "sampling parameters changed" in excinfo.value.message


def test_case_5_anchor_item_added() -> None:
    base = {f"a{i}": 0.6 for i in range(50)}
    added = {**base, "a50": 0.6}
    with pytest.raises(PinViolationError) as excinfo:
        check_anchor_pin(make_anchor_pin(added), make_anchor_pin(base))
    exc = excinfo.value
    assert set(exc.changed) == {"item_set_hash", "baseline_scores_hash", "n"}
    assert "50 -> 51 items" in exc.message
    assert "benchlock rebaseline --reason anchor-set-change" in exc.hint


def test_case_6_anchor_item_removed() -> None:
    base = {f"a{i}": 0.6 for i in range(50)}
    removed = {k: v for k, v in base.items() if k != "a0"}
    with pytest.raises(PinViolationError) as excinfo:
        check_anchor_pin(make_anchor_pin(removed), make_anchor_pin(base))
    assert "50 -> 49 items" in excinfo.value.message
    assert "added or removed" in excinfo.value.message


def test_case_7_anchor_score_edited() -> None:
    base = {f"a{i}": 0.6 for i in range(50)}
    edited = {**base, "a7": 0.9}
    with pytest.raises(PinViolationError) as excinfo:
        check_anchor_pin(make_anchor_pin(edited), make_anchor_pin(base))
    exc = excinfo.value
    # Only the scores moved; membership is untouched, and the message says exactly that.
    assert exc.changed == ("baseline_scores_hash",)
    assert "frozen baseline scores were edited while the item set stayed the same" in exc.message


def test_case_8_scale_changed() -> None:
    with pytest.raises(PinViolationError) as excinfo:
        check_judge_pin(make_judge_pin(scale=(1.0, 10.0)), make_judge_pin())
    assert excinfo.value.changed == ("scale",)
    assert "score scale changed" in excinfo.value.message


# --- supporting properties ----------------------------------------------------------------


def test_provider_change_is_detected() -> None:
    with pytest.raises(PinViolationError):
        check_judge_pin(make_judge_pin(provider="openai"), make_judge_pin())


def test_multiple_simultaneous_changes_are_all_named() -> None:
    with pytest.raises(PinViolationError) as excinfo:
        check_judge_pin(
            make_judge_pin(model="other", params={"temperature": 1.0, "max_tokens": 512}),
            make_judge_pin(),
        )
    assert set(excinfo.value.changed) == {"model", "params_hash"}


def test_anchor_membership_swap_with_same_size_is_distinguished() -> None:
    base = {f"a{i}": 0.6 for i in range(50)}
    swapped = {**{k: v for k, v in base.items() if k != "a0"}, "z0": 0.6}
    with pytest.raises(PinViolationError) as excinfo:
        check_anchor_pin(make_anchor_pin(swapped), make_anchor_pin(base))
    assert "swapped, not added or removed" in excinfo.value.message


def test_anchor_mode_change_is_detected() -> None:
    base = make_anchor_pin()
    with pytest.raises(PinViolationError) as excinfo:
        check_anchor_pin(replace(base, mode=AnchorMode.HUMAN), base)
    assert "anchor mode changed" in excinfo.value.message


def test_pins_serialise_round_trip() -> None:
    judge = make_judge_pin()
    assert JudgePin.from_json(judge.to_json()) == judge
    anchor = make_anchor_pin()
    assert AnchorPin.from_json(anchor.to_json()) == anchor


def test_hashes_are_order_independent_for_item_sets() -> None:
    assert sha256_ids(["b", "a"]) == sha256_ids(["a", "b"])
    assert baseline_scores_hash({"b": 1.0, "a": 0.5}) == baseline_scores_hash({"a": 0.5, "b": 1.0})


def test_baseline_scores_hash_is_precision_stable() -> None:
    # Rounded to 9dp so the pin survives platform float formatting differences,
    # while staying far finer than any judge's resolution.
    assert baseline_scores_hash({"a": 0.1 + 0.2}) == baseline_scores_hash({"a": 0.3})
    assert baseline_scores_hash({"a": 0.3}) != baseline_scores_hash({"a": 0.30001})
