"""Tier 2 step 2: build run-by-run streams from the pooled real judge scores.

**State this plainly wherever the numbers appear:** these streams are *resampled from real
judge scores*, not longitudinally observed. Nobody watched a production pipeline for sixty
runs and recorded what happened. What was observed is how a real judge scores a real item
pool under five configurations; the time axis is constructed by splicing those pools at
known change points.

That construction is what makes Tier 2 affordable, and it is a real limitation. Simulation
gives the statistical power; real judges give the premise; neither alone is a longitudinal
production study, and we have not run one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchlock.model.pins import AnchorMode, AnchorPin, JudgePin, baseline_scores_hash
from benchlock.model.streams import Observation, RunRecord, StreamKind, suite_hash_of

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True, slots=True)
class Scenario:
    """One Tier 2 stream, with its ground truth recorded up front."""

    name: str
    truth: str  # judge | system | both | stable
    #: Which pooled configuration each stream reads from, before and after the change.
    system_before: str
    system_after: str
    anchor_before: str
    anchor_after: str
    change_at: int = 20
    description: str = ""


def scenarios() -> list[Scenario]:
    """The six scenarios, each with known ground truth."""
    return [
        Scenario(
            "judge-version-bump",
            "judge",
            "a-baseline",
            "b-other-snapshot",
            "a-baseline",
            "b-other-snapshot",
            description="the provider rotated the snapshot; the system never changed",
        ),
        Scenario(
            "judge-rubric-change",
            "judge",
            "a-baseline",
            "c-strict-rubric",
            "a-baseline",
            "c-strict-rubric",
            description="a teammate edited the rubric to be stricter",
        ),
        Scenario(
            "judge-parameter-change",
            "judge",
            "a-baseline",
            "d-effort",
            "a-baseline",
            "d-effort",
            description="the judge's reasoning effort was raised from low to high",
        ),
        Scenario(
            "system-regression",
            "system",
            "a-baseline",
            "a-baseline-degraded",
            "a-baseline",
            "a-baseline",
            description="degraded system outputs, judged by an unchanged judge",
        ),
        Scenario(
            "both-moved",
            "both",
            "a-baseline",
            "b-other-snapshot-degraded",
            "a-baseline",
            "b-other-snapshot",
            description="a judge swap and a system regression in the same window",
        ),
        Scenario(
            "drift-free-control",
            "stable",
            "a-baseline",
            "a-baseline",
            "a-baseline",
            "a-baseline",
            description="nothing changed; the control",
        ),
    ]


def _pool_scores(pool: dict[str, Any], key: str) -> dict[str, float]:
    """Pooled scores for a configuration. `-degraded` applies a known system regression.

    The degradation is applied to the *scores*, not by regenerating outputs, because the
    pool is a fixed set of real judge scores. It is a construction, and it is disclosed.
    """
    if key.endswith("-degraded"):
        base = pool["scores"][key.removesuffix("-degraded")]
        return {item: max(0.0, score - 0.10) for item, score in base.items()}
    return dict(pool["scores"][key])


def compose(
    scenario: Scenario,
    pool: dict[str, Any],
    *,
    n_runs: int = 60,
    seed: int = 0,
    items_per_run: int = 150,
    anchor_items: int = 150,
    replicates: int = 5,
) -> tuple[list[RunRecord], list[RunRecord]]:
    """Build (system runs, anchor runs) by sampling from the pool. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    # Only items every pooled configuration scored. `.get(item, default)` would fabricate
    # a score for a missing item, which is precisely the kind of invented number this
    # project exists to make impossible.
    all_items = sorted(set.intersection(*(set(v) for v in pool["scores"].values())))
    # DISJOINT. Both slices used to start at zero, so with the default sizes the anchor
    # set and the system suite were the identical 150 items — the control group was the
    # treatment group. That makes the difference-in-differences vacuous: a judge change
    # moves two copies of one thing, and only the fixed offset subtracted by the
    # `-degraded` scenarios distinguishes the legs at all. The anchor set is defined as
    # items the system under test never touches, so build it that way here too.
    needed = anchor_items + items_per_run
    if len(all_items) < needed:
        raise SystemExit(
            f"pool has {len(all_items)} items scored by every configuration, but a disjoint "
            f"anchor set ({anchor_items}) and system suite ({items_per_run}) need {needed}. "
            f"Re-run bench/real/build_pool.py with more items, or lower the sizes."
        )
    anchor_ids = all_items[:anchor_items]
    system_ids = all_items[anchor_items : anchor_items + items_per_run]
    assert not set(anchor_ids) & set(system_ids)

    replicate_scorings = [dict(s) for s in pool["replicate_scores"]]
    run_means = [float(np.mean([s[i] for i in anchor_ids if i in s])) for s in replicate_scorings]
    per_item = float(
        np.mean(
            [
                float(np.std([s[i] for s in replicate_scorings if i in s], ddof=1))
                for i in anchor_ids
                if all(i in s for s in replicate_scorings)
            ]
        )
    )
    from benchlock.model.pins import NoiseFloor

    noise_floor = NoiseFloor(
        per_item_sd=max(per_item, 1e-6),
        run_mean_sd=max(float(np.std(run_means, ddof=1)), 1e-6),
        replicates=len(replicate_scorings),
        n_items=len(anchor_ids),
    )
    anchor_pin = AnchorPin(
        mode=AnchorMode.FROZEN_SELF,
        item_set_hash=suite_hash_of(anchor_ids),
        baseline_scores_hash=baseline_scores_hash(
            {i: replicate_scorings[0][i] for i in anchor_ids if i in replicate_scorings[0]}
        ),
        n=len(anchor_ids),
        noise_floor=noise_floor,
    )

    def pin_for(config_key: str) -> JudgePin:
        return JudgePin.build(
            provider="pooled",
            model="pooled-real-judge",
            rubric_text=config_key.removesuffix("-degraded"),
            params={"pool": "tier2"},
            scale=(1.0, 5.0),
        )

    system_runs: list[RunRecord] = []
    anchor_runs: list[RunRecord] = []
    for t in range(n_runs):
        changed = t >= scenario.change_at
        system_scores = _pool_scores(
            pool, scenario.system_after if changed else scenario.system_before
        )
        anchor_scores = _pool_scores(
            pool, scenario.anchor_after if changed else scenario.anchor_before
        )
        # Run-to-run variation comes from resampling which pooled replicate each item
        # takes, which is why the noise here is the judge's own measured noise.
        jitter = rng.normal(0.0, noise_floor.per_item_sd / 4.0, len(system_ids))

        system_obs = tuple(
            Observation(
                item_id=item,
                score=float(np.clip(system_scores[item] + jitter[k], 0.0, 1.0)),
                raw_score=1.0 + 4.0 * float(np.clip(system_scores[item], 0.0, 1.0)),
                scale=(1.0, 5.0),
            )
            for k, item in enumerate(system_ids)
        )
        anchor_jitter = rng.normal(0.0, noise_floor.per_item_sd / 4.0, len(anchor_ids))
        anchor_obs = tuple(
            Observation(
                item_id=item,
                score=float(np.clip(anchor_scores[item] + anchor_jitter[k], 0.0, 1.0)),
                raw_score=1.0 + 4.0 * float(np.clip(anchor_scores[item], 0.0, 1.0)),
                scale=(1.0, 5.0),
            )
            for k, item in enumerate(anchor_ids)
        )
        # The judge pin is deliberately held constant across the splice: these scenarios
        # are about *silent* provider-side change, which is the case the anchor set exists
        # for. A declared change would be caught by the pin check without any statistics.
        pin = pin_for(scenario.system_before)
        system_runs.append(
            RunRecord(
                run_id=f"{scenario.name}-sys-{t:04d}",
                run_index=t,
                kind=StreamKind.SYSTEM,
                observations=system_obs,
                suite_hash=suite_hash_of(o.item_id for o in system_obs),
                judge_pin=pin,
                anchor_pin=None,
            )
        )
        anchor_runs.append(
            RunRecord(
                run_id=f"{scenario.name}-anc-{t:04d}",
                run_index=t,
                kind=StreamKind.ANCHOR,
                observations=anchor_obs,
                suite_hash=suite_hash_of(o.item_id for o in anchor_obs),
                judge_pin=pin,
                anchor_pin=anchor_pin,
            )
        )
    del replicates
    return system_runs, anchor_runs


def load_pool(path: Path | None = None) -> dict[str, Any]:
    target = path or ROOT / "bench" / "results" / "real-pool.json"
    if not target.exists():
        raise SystemExit(
            f"no scored pool at {target}. Build one first:\n"
            "  uv run python bench/real/build_pool.py --items 400"
        )
    return json.loads(target.read_text())


METHODOLOGY = (
    "Tier 2 streams are **constructed by resampling pooled real judge scores**, not "
    "observed longitudinally. A fixed item pool was scored once under each judge "
    "configuration; the time axis is built by splicing those pools at known change "
    "points. Two further constructions must be stated plainly. **The judge-change "
    "scenarios are real**: they splice scores that a real judge actually produced under "
    "two configurations, on an anchor set that is **disjoint from the system suite** — 150 "
    "items each, from different halves of the pool, so the control group really is made of "
    "items the system under test never touches. **The system-regression scenarios are "
    "not real**: no degraded system "
    "was built, and the 'regression' is a fixed 0.10 subtracted from the baseline judge's "
    "real scores (`compose._pool_scores`, the `-degraded` suffix). That tests whether the "
    "attribution machinery separates a subtracted shift from a real judge change; it does "
    "not test whether a real judge notices a real system regression. Simulation supplies "
    "the statistical power, real judges supply the premise, and neither is a longitudinal "
    "production study. We have not run one."
)


def scenario_table() -> Sequence[dict[str, str]]:
    return [{"name": s.name, "truth": s.truth, "description": s.description} for s in scenarios()]
