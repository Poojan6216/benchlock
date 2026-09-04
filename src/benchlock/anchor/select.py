"""Choosing which items become the anchor set.

Anchors must resemble the eval distribution. If they do not, a judge change confined to a
region they do not cover is invisible, the system stream moves, and the verdict is
`SYSTEM` — a confident, wrong rollback. That is the fundamental limitation of the whole
design (Phase 7.1 measures how bad it gets), and stratified selection is the cheapest
thing that makes it less likely.

Selection is **deterministic given a seed**, and the seed is pinned in the config, so an
anchor set can be reconstructed and audited rather than merely trusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from benchlock.config import SelectionKind

#: Number of score strata. Deciles are the spec's choice and are fine for suites of a few
#: hundred items; finer strata would leave several empty.
N_DECILES = 10


@dataclass(frozen=True, slots=True)
class Candidate:
    """One eval item that could become an anchor."""

    item_id: str
    score: float  # its baseline score, normalised to [0, 1]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Selection:
    chosen: tuple[str, ...]
    seed: int
    kind: SelectionKind
    #: How many items came from each (decile, tag) stratum, for the audit trail.
    strata: tuple[tuple[str, int], ...]

    def __len__(self) -> int:
        return len(self.chosen)


def decile_of(score: float) -> int:
    """Which score decile a value falls in. 1.0 belongs to the top decile, not an 11th."""
    return min(int(score * N_DECILES), N_DECILES - 1)


def _stratum_key(candidate: Candidate) -> str:
    tag = candidate.tags[0] if candidate.tags else "-"
    return f"d{decile_of(candidate.score)}/{tag}"


def select_anchors(
    candidates: Sequence[Candidate],
    n: int,
    *,
    seed: int = 0,
    kind: SelectionKind = SelectionKind.STRATIFIED,
) -> Selection:
    """Choose `n` anchor items. Deterministic in `seed`.

    Stratified selection allocates proportionally to each (decile, tag) stratum, then
    fills any shortfall — strata that had fewer items than their quota — from what is
    left, so the requested size is met exactly whenever the suite is large enough.
    """
    if n < 1:
        raise ValueError(f"anchor size must be at least 1, got {n}")
    if not candidates:
        raise ValueError("cannot select anchors from an empty suite")
    if n > len(candidates):
        raise ValueError(
            f"asked for {n} anchor items but the suite has only {len(candidates)}. "
            "Either grow the eval suite or lower `anchor.n` — and note that `benchlock plan` "
            "sized it for a reason, so a smaller anchor set means a larger detectable shift"
        )

    rng = np.random.default_rng(seed)
    ordered = sorted(candidates, key=lambda c: c.item_id)  # arrival order must not matter

    if kind is SelectionKind.RANDOM:
        picked = [ordered[i] for i in rng.permutation(len(ordered))[:n]]
        return Selection(
            chosen=tuple(sorted(c.item_id for c in picked)),
            seed=seed,
            kind=kind,
            strata=_count_strata(picked),
        )

    # --- stratified ---
    buckets: dict[str, list[Candidate]] = {}
    for candidate in ordered:
        buckets.setdefault(_stratum_key(candidate), []).append(candidate)

    # Every occupied stratum gets at least one item before anything is allocated
    # proportionally. Without that floor, a rare stratum's quota rounds to zero and the
    # stratum vanishes from the anchor set entirely — which is precisely the hole a
    # region-confined judge change hides in (Phase 7.1). The floor applies while there is
    # room for it; past that, proportionality takes over.
    guarantee = 1 if n >= len(buckets) else 0

    chosen: list[Candidate] = []
    leftovers: list[Candidate] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        shuffled = [bucket[i] for i in rng.permutation(len(bucket))]
        proportional = int(len(bucket) * n / len(ordered))
        quota = min(len(bucket), max(proportional, guarantee))
        chosen.extend(shuffled[:quota])
        leftovers.extend(shuffled[quota:])

    # The floor can overshoot n when there are many sparse strata; drop back down from the
    # largest strata first so the guarantee survives the trim.
    if len(chosen) > n:
        counts: dict[str, int] = {}
        for candidate in chosen:
            counts[_stratum_key(candidate)] = counts.get(_stratum_key(candidate), 0) + 1
        trimmed: list[Candidate] = []
        for candidate in sorted(chosen, key=lambda c: (-counts[_stratum_key(c)], c.item_id))[::-1]:
            if len(trimmed) < n:
                trimmed.append(candidate)
        chosen = trimmed

    shortfall = n - len(chosen)
    if shortfall > 0:
        order = rng.permutation(len(leftovers))
        chosen.extend(leftovers[i] for i in order[:shortfall])

    return Selection(
        chosen=tuple(sorted(c.item_id for c in chosen[:n])),
        seed=seed,
        kind=kind,
        strata=_count_strata(chosen[:n]),
    )


def _count_strata(items: Sequence[Candidate]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for item in items:
        counts[_stratum_key(item)] = counts.get(_stratum_key(item), 0) + 1
    return tuple(sorted(counts.items()))
