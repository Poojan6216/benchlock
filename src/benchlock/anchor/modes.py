"""The three ways to build a control group, and why the default needs no labels.

**`frozen-self` (the default).** Freeze a set of ``(item_id, system_output)`` pairs and
snapshot the judge's own scores on them. Thereafter re-score the same frozen pairs and test
for divergence from the snapshot beyond the measured noise floor.

The observation that makes this work: **attribution needs judge *stability*, not judge
*validity*.** We are not asking whether the judge is right. We are asking whether it is the
same judge it was last week. A frozen snapshot of its own scores answers that exactly, and
costs nothing to produce — which is the difference between a tool a team adopts this
afternoon and one that needs a multi-week labelling project first.

The cost of that trade is real and goes in the README, not an appendix: a judge that was
always wrong stays consistently wrong and correctly reads as stable. Benchlock does not
claim your evals are correct. It claims to tell you which of two things moved.

**`human`** adds optional gold labels, which buys a judge-versus-human agreement series on
top of stability — validity as a secondary signal. Never required.

**`replicate`** re-scores a subsample of the *current* run with a pinned baseline snapshot,
making the judge itself the control rather than a frozen item set. Requires a provider that
offers dated snapshots, and refuses where one is not available.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchlock.anchor.noisefloor import NoiseFloorReport, estimate_noise_floor
from benchlock.judge.base import JudgeAdapter, JudgeRequest
from benchlock.model.pins import AnchorMode, AnchorPin, JudgePin, baseline_scores_hash
from benchlock.model.streams import suite_hash_of

#: Where the frozen anchor pairs live. This file holds eval *content* — the system outputs
#: that must be re-scored every run — so unlike the ledger it is not safe to share. It sits
#: under .benchlock/, which the generated .gitignore excludes.
DEFAULT_ANCHOR_STORE = Path(".benchlock/anchors.jsonl")


class AnchorModeError(Exception):
    def __init__(self, message: str, hint: str = "") -> None:
        self.message = message
        self.hint = hint
        super().__init__(f"{message}" + (f"\n  fix: {hint}" if hint else ""))


@dataclass(frozen=True, slots=True)
class AnchorItem:
    """One frozen pair. The system under test never regenerates these outputs."""

    item_id: str
    prompt_input: str
    output: str
    tags: tuple[str, ...] = ()
    #: Optional gold label, only used in `human` mode.
    gold: float | None = None


@dataclass(frozen=True, slots=True)
class FrozenAnchor:
    """The result of `benchlock baseline`: the pinned control group and its noise floor."""

    pin: AnchorPin
    baseline_scores: Mapping[str, float]
    noise: NoiseFloorReport
    judge_pin: JudgePin
    mode: AnchorMode
    #: The K individual replicate scorings. Recorded to the ledger as the first K anchor
    #: runs so `decide()` and `replay` can rebuild the snapshot from the ledger alone,
    #: rather than trusting a number written once and never checkable again.
    replicate_scorings: tuple[Mapping[str, float], ...] = ()

    def warnings(self) -> tuple[str, ...]:
        return self.noise.warnings()


def freeze(
    items: Sequence[AnchorItem],
    judge: JudgeAdapter,
    *,
    replicates: int = 5,
    mode: AnchorMode = AnchorMode.FROZEN_SELF,
    nonce_prefix: str = "",
) -> FrozenAnchor:
    """Score the anchor set K times and pin the result.

    The K replicates do two jobs at once: their mean is the frozen baseline, and their
    spread is the noise floor every later test is measured against. Both come from the
    same calls, so measuring the floor costs nothing beyond the replicates themselves.
    """
    if not items:
        raise AnchorModeError(
            "cannot freeze an empty anchor set",
            "select anchors from your eval suite first; `benchlock plan` will size the set",
        )
    if replicates < 2:
        raise AnchorModeError(
            f"a baseline needs at least K=2 replicates to measure a noise floor, got {replicates}",
            "set `anchor.noise_replicates` to 5 (the default)",
        )
    if mode is AnchorMode.HUMAN and any(item.gold is None for item in items):
        missing = sum(1 for item in items if item.gold is None)
        raise AnchorModeError(
            f"`human` mode needs a gold label on every anchor item; {missing} are missing",
            "supply labels via `anchor.labels`, or use the default `frozen-self` mode, "
            "which needs none",
        )

    description = judge.describe()
    lo, hi = description.scale
    scorings: list[dict[str, float]] = []
    for k in range(replicates):
        requests = [
            JudgeRequest(
                item_id=item.item_id,
                prompt_input=item.prompt_input,
                output=item.output,
                nonce=f"{nonce_prefix}{k}" if nonce_prefix else "",
            )
            for item in items
        ]
        results = judge.score(requests)
        scoring: dict[str, float] = {}
        for result in results:
            if not lo <= result.raw_score <= hi:
                raise AnchorModeError(
                    f"judge returned {result.raw_score} for item {result.item_id!r}, outside "
                    f"the declared score_scale [{lo}, {hi}]",
                    "either the rubric emits a wider range than declared, or the judge is "
                    "ignoring the rubric. Benchlock will not clamp",
                )
            scoring[result.item_id] = (result.raw_score - lo) / (hi - lo)
        scorings.append(scoring)

    noise = estimate_noise_floor(scorings)
    baseline = {item.item_id: sum(s[item.item_id] for s in scorings) / replicates for item in items}
    pin = AnchorPin(
        mode=mode,
        item_set_hash=suite_hash_of(baseline),
        baseline_scores_hash=baseline_scores_hash(baseline),
        n=len(baseline),
        noise_floor=noise.floor,
    )
    return FrozenAnchor(
        pin=pin,
        baseline_scores=baseline,
        noise=noise,
        judge_pin=judge.pin(),
        mode=mode,
        replicate_scorings=tuple(scorings),
    )


def rescore(
    items: Sequence[AnchorItem],
    judge: JudgeAdapter,
    *,
    nonce: str = "",
) -> dict[str, float]:
    """Re-score the frozen pairs for one run. Returns normalised scores."""
    lo, hi = judge.describe().scale
    requests = [
        JudgeRequest(
            item_id=item.item_id,
            prompt_input=item.prompt_input,
            output=item.output,
            nonce=nonce,
        )
        for item in items
    ]
    out: dict[str, float] = {}
    for result in judge.score(requests):
        if not lo <= result.raw_score <= hi:
            raise AnchorModeError(
                f"judge returned {result.raw_score} for item {result.item_id!r}, outside the "
                f"declared score_scale [{lo}, {hi}]",
                "benchlock will not clamp a score into its declared range",
            )
        out[result.item_id] = (result.raw_score - lo) / (hi - lo)
    return out


def check_mode_supported(mode: AnchorMode, judge: JudgeAdapter) -> None:
    """Phase 3.3: `replicate` needs a provider with pinnable dated snapshots."""
    if mode is not AnchorMode.REPLICATE:
        return
    description = judge.describe()
    if not description.supports_pinned_snapshots:
        raise AnchorModeError(
            f"`replicate` anchor mode needs a provider that offers pinned dated snapshots, "
            f"and {description.provider!r} does not expose one for {description.model!r}",
            "use `mode: frozen-self`, which needs neither pinned snapshots nor labels, or "
            "switch to a provider that publishes dated model snapshots",
        )


def agreement_series(scores: Mapping[str, float], items: Sequence[AnchorItem]) -> tuple[float, int]:
    """`human` mode's extra signal: mean |judge - gold| and how many items had labels.

    This measures judge *validity*, which is a different question from the stability the
    verdict rests on. It is reported alongside, never folded into the decision.
    """
    pairs = [
        (scores[i.item_id], i.gold) for i in items if i.gold is not None and i.item_id in scores
    ]
    if not pairs:
        return 0.0, 0
    total = sum(abs(score - gold) for score, gold in pairs)
    return total / len(pairs), len(pairs)


# ---------------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------------


def save_anchors(items: Sequence[AnchorItem], path: Path = DEFAULT_ANCHOR_STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "item_id": i.item_id,
                    "prompt_input": i.prompt_input,
                    "output": i.output,
                    "tags": list(i.tags),
                    **({"gold": i.gold} if i.gold is not None else {}),
                },
                sort_keys=True,
            )
            for i in items
        )
        + "\n",
        encoding="utf-8",
    )


def load_anchors(path: Path = DEFAULT_ANCHOR_STORE) -> list[AnchorItem]:
    if not path.exists():
        raise AnchorModeError(
            f"no frozen anchor set at {path}",
            "run `benchlock baseline` to freeze one and measure the judge's noise floor",
        )
    items: list[AnchorItem] = []
    # The store is written non-atomically, so an interrupted `baseline` leaves a half
    # line behind. Every caller here catches AnchorModeError and nothing else, so a bare
    # json.loads would escape as a traceback — and it would do so from `observe
    # --rescore-anchors`, i.e. AFTER the system run has already been committed to an
    # append-only ledger. The user needs to be told which file and which line.
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnchorModeError(
                f"the frozen anchor store at {path} is corrupt: line {line_no} is not "
                f"valid JSON ({exc.msg})",
                "this usually means a `benchlock baseline` was interrupted while writing. "
                "Re-run `benchlock baseline --anchors <your-suite>` to freeze it again, "
                "and start a new epoch with `benchlock rebaseline` if the set has changed",
            ) from exc
        if not isinstance(data, dict) or "item_id" not in data:
            raise AnchorModeError(
                f"the frozen anchor store at {path} is corrupt: line {line_no} has no `item_id`",
                "every line must be a JSON object with at least an `item_id`; re-run "
                "`benchlock baseline --anchors <your-suite>` to rewrite the store",
            )
        items.append(
            AnchorItem(
                item_id=data["item_id"],
                prompt_input=data.get("prompt_input", ""),
                output=data.get("output", ""),
                tags=tuple(data.get("tags", ())),
                gold=data.get("gold"),
            )
        )
    return items
