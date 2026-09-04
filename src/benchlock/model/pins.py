"""Pinned identity of the things that, if they change, invalidate a comparison.

Hard Rule 8: a change to a pin without an explicit, logged rebaseline is an ERROR,
not a warning. The hashing here is what makes that detectable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class AnchorMode(StrEnum):
    """How the anchor (control) stream is constructed."""

    FROZEN_SELF = "frozen-self"  # snapshot the judge's own scores. no human labels.
    HUMAN = "human"  # gold labels on anchor items (optional, adds validity signal)
    REPLICATE = "replicate"  # re-score a subsample with a pinned dated snapshot


def sha256_text(text: str) -> str:
    """SHA-256 of text, encoded UTF-8. Byte-exact: whitespace changes the hash.

    Whitespace matters because whitespace changes prompts (Phase 0.5 verify case 2).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """SHA-256 of a canonical JSON encoding: sorted keys, no insignificant whitespace."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))


def sha256_ids(item_ids: Iterable[object]) -> str:
    """SHA-256 of a *sorted* collection of item ids. Order of arrival must not matter."""
    return sha256_json(sorted(str(i) for i in item_ids))


@dataclass(frozen=True, slots=True)
class NoiseFloor:
    """Within-judge variability, measured from K replicates at baseline (Phase 1.5).

    Lives in `model/` rather than `anchor/noisefloor.py` so that the model layer has no
    dependencies; `anchor/noisefloor.py` is the *estimator* that produces one of these.
    All downstream tests are against this floor, never against zero.
    """

    per_item_sd: float  # SD of a single item's score across identical calls
    run_mean_sd: float  # SD of the run-level mean across identical calls
    replicates: int  # K
    n_items: int  # anchor items the estimate was taken over

    def __post_init__(self) -> None:
        if self.per_item_sd < 0.0 or self.run_mean_sd < 0.0:
            raise ValueError("noise floor standard deviations must be non-negative")
        if self.replicates < 2:
            raise ValueError("a noise floor needs at least K=2 replicates to be estimable")
        if self.n_items < 1:
            raise ValueError("a noise floor needs at least one item")


@dataclass(frozen=True, slots=True)
class JudgePin:
    """Everything about the judge that, if changed, invalidates comparison.

    Hard Rule 8: a change here without an explicit rebaseline is an ERROR.
    """

    provider: str
    model: str  # the exact dated snapshot where the provider offers one
    rubric_hash: str  # SHA-256 of the full rubric/system prompt
    params_hash: str  # temperature, top_p, max_tokens, response format, seed
    scale: tuple[float, float]

    def differs_from(self, other: JudgePin) -> tuple[str, ...]:
        """Names of the fields that changed, for the verdict message. Empty if identical."""
        changed = [
            name
            for name in ("provider", "model", "rubric_hash", "params_hash", "scale")
            if getattr(self, name) != getattr(other, name)
        ]
        return tuple(changed)

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "rubric_hash": self.rubric_hash,
            "params_hash": self.params_hash,
            "scale": [self.scale[0], self.scale[1]],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> JudgePin:
        scale = data["scale"]
        return cls(
            provider=str(data["provider"]),
            model=str(data["model"]),
            rubric_hash=str(data["rubric_hash"]),
            params_hash=str(data["params_hash"]),
            scale=(float(scale[0]), float(scale[1])),
        )

    @classmethod
    def build(
        cls,
        *,
        provider: str,
        model: str,
        rubric_text: str,
        params: dict[str, Any],
        scale: tuple[float, float],
    ) -> JudgePin:
        """Hash the rubric text and params into a pin.

        The rubric is hashed byte-exactly: a whitespace-only edit changes the prompt,
        so it must change the pin (Phase 0.5 verify case 2).
        """
        return cls(
            provider=provider,
            model=model,
            rubric_hash=sha256_text(rubric_text),
            params_hash=sha256_json(params),
            scale=scale,
        )


@dataclass(frozen=True, slots=True)
class AnchorPin:
    """The frozen control set. Any change to its membership or baseline invalidates it."""

    mode: AnchorMode
    item_set_hash: str  # SHA-256 of sorted anchor item_ids
    baseline_scores_hash: str
    n: int
    noise_floor: NoiseFloor

    def differs_from(self, other: AnchorPin) -> tuple[str, ...]:
        changed = [
            name
            for name in ("mode", "item_set_hash", "baseline_scores_hash", "n")
            if getattr(self, name) != getattr(other, name)
        ]
        return tuple(changed)

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "item_set_hash": self.item_set_hash,
            "baseline_scores_hash": self.baseline_scores_hash,
            "n": self.n,
            "noise_floor": {
                "per_item_sd": self.noise_floor.per_item_sd,
                "run_mean_sd": self.noise_floor.run_mean_sd,
                "replicates": self.noise_floor.replicates,
                "n_items": self.noise_floor.n_items,
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AnchorPin:
        nf = data["noise_floor"]
        return cls(
            mode=AnchorMode(data["mode"]),
            item_set_hash=str(data["item_set_hash"]),
            baseline_scores_hash=str(data["baseline_scores_hash"]),
            n=int(data["n"]),
            noise_floor=NoiseFloor(
                per_item_sd=float(nf["per_item_sd"]),
                run_mean_sd=float(nf["run_mean_sd"]),
                replicates=int(nf["replicates"]),
                n_items=int(nf["n_items"]),
            ),
        )


def baseline_scores_hash(scores: Mapping[str, float]) -> str:
    """Hash of the frozen baseline scores, rounded to a stable precision.

    Rounded because float formatting differs across platforms and the pin must be
    reproducible; 9 decimal places is far finer than any judge's resolution.
    """
    return sha256_json({k: round(float(v), 9) for k, v in sorted(scores.items())})


# ---------------------------------------------------------------------------------------
# Violation detection (Hard Rule 8)
# ---------------------------------------------------------------------------------------

#: What each pin field means to someone reading an error at 2am.
_FIELD_MEANING: dict[str, str] = {
    "provider": "the judge provider changed",
    "model": "the judge model snapshot changed",
    "rubric_hash": (
        "the rubric text changed — including whitespace, because whitespace changes prompts"
    ),
    "params_hash": "the judge sampling parameters changed (temperature, top_p, max_tokens, seed)",
    "scale": "the declared score scale changed",
    "mode": "the anchor mode changed",
    "item_set_hash": "the anchor item set changed",
    "baseline_scores_hash": "the frozen baseline anchor scores changed",
    "n": "the anchor set size changed",
}


class PinViolationError(Exception):
    """A pinned value moved without a logged rebaseline.

    This is an error, not a warning (Hard Rule 8). Comparing scores across a judge or
    anchor change is comparing two different measurements and calling the difference a
    result.
    """

    def __init__(
        self,
        which: str,
        changed: Sequence[str],
        detail: Sequence[str] = (),
        *,
        rebaseline_reason: str = "",
    ) -> None:
        self.which = which
        self.changed = tuple(changed)
        self.detail = tuple(detail)
        reason = rebaseline_reason or (
            "judge-version-change" if which == "judge" else "anchor-set-change"
        )
        self.message = (
            f"{which} pin changed without a rebaseline: {', '.join(self.changed)}\n"
            + "\n".join(f"  - {d}" for d in self.detail)
        )
        self.hint = (
            "scores before and after this change are not comparable. If the change was "
            f"intentional, record it:\n    benchlock rebaseline --reason {reason}"
        )
        super().__init__(f"{self.message}\n  fix: {self.hint}")


def _fmt(value: object) -> str:
    """Show plaintext in full; abbreviate hex digests.

    The dated suffix of a model snapshot is the whole point of the message, so plaintext
    is never truncated. A 64-char SHA-256 carries no information past its prefix.
    """
    text = str(value)
    is_digest = len(text) == 64 and all(c in "0123456789abcdef" for c in text)
    return f"{text[:12]}…" if is_digest else text


def check_judge_pin(current: JudgePin, pinned: JudgePin) -> None:
    """Raise if the judge moved. No-op when the pins are identical."""
    changed = current.differs_from(pinned)
    if not changed:
        return
    detail = [
        f"{_FIELD_MEANING[field]}: {_fmt(getattr(pinned, field))} -> "
        f"{_fmt(getattr(current, field))}"
        for field in changed
    ]
    raise PinViolationError("judge", changed, detail)


def check_anchor_pin(current: AnchorPin, pinned: AnchorPin) -> None:
    """Raise if the anchor set moved. Distinguishes membership from baseline edits."""
    changed = current.differs_from(pinned)
    if not changed:
        return
    detail: list[str] = []
    if "n" in changed:
        detail.append(f"the anchor set size changed: {pinned.n} -> {current.n} items")
    if "item_set_hash" in changed and "n" not in changed:
        detail.append(
            f"anchor membership changed with the same size ({current.n} items): "
            "an item was swapped, not added or removed"
        )
    elif "item_set_hash" in changed:
        detail.append("the anchor item set changed: items were added or removed")
    if "baseline_scores_hash" in changed and "item_set_hash" not in changed:
        detail.append("the frozen baseline scores were edited while the item set stayed the same")
    elif "baseline_scores_hash" in changed:
        detail.append("the frozen baseline scores changed, as they must when membership does")
    if "mode" in changed:
        detail.append(f"the anchor mode changed: {pinned.mode.value} -> {current.mode.value}")
    raise PinViolationError("anchor", changed, detail)
