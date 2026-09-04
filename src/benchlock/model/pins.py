"""Pinned identity of the things that, if they change, invalidate a comparison.

Hard Rule 8: a change to a pin without an explicit, logged rebaseline is an ERROR,
not a warning. The hashing here is what makes that detectable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
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
