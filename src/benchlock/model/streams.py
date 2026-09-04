"""The two score streams: the system under test, and the frozen anchor set.

The whole identification argument lives in the distinction between these two enum
members. The system under test never touches ANCHOR items, so if the anchor score
moves, only the judge can have moved it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from benchlock.model.pins import AnchorPin, JudgePin, sha256_ids


class StreamKind(StrEnum):
    SYSTEM = "system"  # scores of the system under test
    ANCHOR = "anchor"  # scores of the frozen anchor set. the system never touches these.


@dataclass(frozen=True, slots=True)
class Observation:
    """One judged item.

    Scores are normalised to [0, 1] at ingest; the original scale and bounds are recorded
    so the normalisation is reversible and auditable.
    """

    item_id: str
    score: float  # in [0, 1]
    raw_score: float
    scale: tuple[float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError(f"item {self.item_id!r}: normalised score is not finite")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"item {self.item_id!r}: normalised score {self.score} is outside [0, 1]"
            )

    @classmethod
    def normalised(cls, item_id: str, raw_score: float, scale: tuple[float, float]) -> Observation:
        """Normalise a raw score onto [0, 1]. Out-of-range input is an error, not a clamp."""
        lo, hi = scale
        if not hi > lo:
            raise ValueError(f"score scale must be [low, high] with high > low, got [{lo}, {hi}]")
        if not math.isfinite(raw_score):
            raise ValueError(f"item {item_id!r}: score {raw_score!r} is not a finite number")
        if not lo <= raw_score <= hi:
            raise ValueError(
                f"item {item_id!r}: score {raw_score} is outside the declared "
                f"score_scale [{lo}, {hi}]"
            )
        return cls(
            item_id=item_id, score=(raw_score - lo) / (hi - lo), raw_score=raw_score, scale=scale
        )

    def denormalised(self) -> float:
        """Invert the normalisation. Round-trips to `raw_score` up to float precision."""
        lo, hi = self.scale
        return lo + self.score * (hi - lo)

    def to_json(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "score": self.score, "raw": self.raw_score}


def suite_hash_of(item_ids: Iterable[str]) -> str:
    """SHA-256 of the sorted item ids. Lattice rule 2 compares this between runs."""
    return sha256_ids(item_ids)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One CI run. This is the unit of the stream."""

    run_id: str  # ULID
    run_index: int  # monotonic, 0-based
    kind: StreamKind
    observations: tuple[Observation, ...]
    suite_hash: str  # SHA-256 of the sorted item_ids. Hard Rule 6 checks this.
    judge_pin: JudgePin
    anchor_pin: AnchorPin | None  # None for SYSTEM streams
    #: Baseline epoch. Incremented by `benchlock rebaseline`; verdicts never compare
    #: across an epoch boundary (Hard Rule 8).
    epoch: int = 0

    def __post_init__(self) -> None:
        if self.run_index < 0:
            raise ValueError(f"run_index must be >= 0, got {self.run_index}")
        if not self.observations:
            raise ValueError(f"run {self.run_id!r} has no observations")

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def mean(self) -> float:
        """Mean normalised score for this run. The per-run summary the streams monitor."""
        return sum(o.score for o in self.observations) / len(self.observations)

    def scores(self) -> tuple[float, ...]:
        return tuple(o.score for o in self.observations)

    def by_item(self) -> dict[str, float]:
        return {o.item_id: o.score for o in self.observations}


def run_means(runs: Sequence[RunRecord]) -> tuple[float, ...]:
    return tuple(r.mean for r in runs)
