"""How much of the eval distribution the anchor set actually covers.

**Coverage is measured and reported. It is never guaranteed**, and the output says so.
An anchor set that covers 90% of the score range and every tag can still miss a judge
change confined to some other axis nobody thought to tag — response length, language,
whether the answer contains code. Phase 7.1 measures the damage as a function of the
coverage number, and the warning threshold here is set from that measurement rather than
chosen to look reassuring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from benchlock.anchor.select import Candidate, decile_of

#: Below this, the anchor set is missing enough of the suite that a region-confined judge
#: change is likely to be invisible. Calibrated in Phase 7.1.
COVERAGE_WARN_THRESHOLD = 0.7


@dataclass(frozen=True, slots=True)
class Coverage:
    """What fraction of the suite's shape the anchor set reproduces."""

    decile_coverage: float  # fraction of occupied score deciles represented
    tag_coverage: float  # fraction of tags represented
    missing_deciles: tuple[int, ...]
    missing_tags: tuple[str, ...]
    #: Largest absolute difference between a suite stratum's share and the anchors'.
    max_marginal_gap: float
    anchor_n: int
    suite_n: int

    @property
    def adequate(self) -> bool:
        return (
            min(self.decile_coverage, self.tag_coverage) >= COVERAGE_WARN_THRESHOLD
            and not self.missing_tags
        )

    def warnings(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.missing_tags:
            out.append(
                f"the anchor set contains no items tagged "
                f"{', '.join(repr(t) for t in self.missing_tags)}. A judge change confined "
                "to those items would be invisible, and benchlock would attribute the "
                "resulting movement to your system"
            )
        if self.missing_deciles:
            deciles = ", ".join(str(d) for d in self.missing_deciles)
            out.append(
                f"score deciles {deciles} are present in the suite but not in the anchor "
                "set; a judge that became stricter only about items in that range would "
                "not be caught"
            )
        if self.max_marginal_gap > 0.15:
            out.append(
                f"the anchor set's composition differs from the suite's by up to "
                f"{self.max_marginal_gap:.0%} in some stratum; re-run selection with "
                "`selection: stratified` or a larger `anchor.n`"
            )
        return tuple(out)

    def to_json(self) -> dict[str, object]:
        return {
            "decile_coverage": self.decile_coverage,
            "tag_coverage": self.tag_coverage,
            "missing_deciles": list(self.missing_deciles),
            "missing_tags": list(self.missing_tags),
            "max_marginal_gap": self.max_marginal_gap,
            "anchor_n": self.anchor_n,
            "suite_n": self.suite_n,
            "adequate": self.adequate,
            "measured_not_guaranteed": True,
        }


def measure_coverage(suite: Sequence[Candidate], anchor_ids: Sequence[str]) -> Coverage:
    """Compare the anchor set's marginals against the suite's."""
    if not suite:
        raise ValueError("cannot measure coverage against an empty suite")
    chosen = set(anchor_ids)
    anchors = [c for c in suite if c.item_id in chosen]
    if not anchors:
        raise ValueError("none of the anchor ids appear in the suite")

    suite_deciles = {decile_of(c.score) for c in suite}
    anchor_deciles = {decile_of(c.score) for c in anchors}
    suite_tags = {t for c in suite for t in c.tags}
    anchor_tags = {t for c in anchors for t in c.tags}

    gaps = [
        abs(
            sum(1 for c in suite if decile_of(c.score) == d) / len(suite)
            - sum(1 for c in anchors if decile_of(c.score) == d) / len(anchors)
        )
        for d in sorted(suite_deciles)
    ]
    gaps += [
        abs(
            sum(1 for c in suite if tag in c.tags) / len(suite)
            - sum(1 for c in anchors if tag in c.tags) / len(anchors)
        )
        for tag in sorted(suite_tags)
    ]

    return Coverage(
        decile_coverage=len(anchor_deciles & suite_deciles) / len(suite_deciles),
        tag_coverage=(len(anchor_tags & suite_tags) / len(suite_tags)) if suite_tags else 1.0,
        missing_deciles=tuple(sorted(suite_deciles - anchor_deciles)),
        missing_tags=tuple(sorted(suite_tags - anchor_tags)),
        max_marginal_gap=max(gaps) if gaps else 0.0,
        anchor_n=len(anchors),
        suite_n=len(suite),
    )
