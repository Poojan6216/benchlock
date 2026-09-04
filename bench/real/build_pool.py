"""Tier 2 step 1: score a fixed item pool once under each judge configuration.

This is what makes real judges affordable. Re-running a whole pipeline per timepoint would
be tens of thousands of calls; instead the pool is scored **once** under each of five judge
configurations, plus K replicates on the anchor subset for the noise floor. That is roughly
2,000-2,500 calls in total, and every stream in Tier 2 is then composed by sampling from
those pooled scores.

    uv run python bench/real/build_pool.py --items 400 --replicates 5

**Hard Rule 9 applies here too.** The committed pool holds scores and item-id hashes. The
`(input, output)` content is reconstructed from a public dataset by `loader.py` using a
committed manifest of ids, and never checked in.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bench.real.loader import PoolItem, load_items
from benchlock.judge.base import JudgeAdapter, JudgeRequest
from benchlock.judge.cache import ScoreCache
from benchlock.ledger.log import hash_item_id

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "bench" / "results"

BASE_RUBRIC = """\
You are grading an assistant's answer.

Score it from 1 to 5 for helpfulness and factual accuracy:
5 = fully correct and directly useful
4 = correct with minor gaps
3 = partially correct
2 = mostly wrong or unhelpful
1 = wrong or irrelevant

Reply with the number and nothing else.
"""

STRICT_RUBRIC = """\
You are grading an assistant's answer to a strict standard.

Score it from 1 to 5. Reserve 5 for answers that are complete, precise and free of any
hedging or unsupported claims. Penalise vagueness heavily. An answer that is merely
acceptable scores 3.

Reply with the number and nothing else.
"""


@dataclass(frozen=True, slots=True)
class JudgeConfigSpec:
    """One of the five configurations the pool is scored under."""

    key: str
    provider: str
    model: str
    rubric: str
    params: dict[str, Any]
    why: str


def default_configs() -> list[JudgeConfigSpec]:
    """(a) baseline, (b) a different snapshot, (c) stricter rubric, (d) different
    temperature, (e) a different provider entirely."""
    return [
        JudgeConfigSpec(
            "a-baseline",
            "anthropic",
            "claude-sonnet-4-5-20250929",
            BASE_RUBRIC,
            {"temperature": 0.0, "max_tokens": 16},
            "the baseline snapshot",
        ),
        JudgeConfigSpec(
            "b-other-snapshot",
            "anthropic",
            "claude-haiku-4-5-20251001",
            BASE_RUBRIC,
            {"temperature": 0.0, "max_tokens": 16},
            "a different snapshot from the same provider",
        ),
        JudgeConfigSpec(
            "c-strict-rubric",
            "anthropic",
            "claude-sonnet-4-5-20250929",
            STRICT_RUBRIC,
            {"temperature": 0.0, "max_tokens": 16},
            "same model, stricter rubric",
        ),
        JudgeConfigSpec(
            "d-temperature",
            "anthropic",
            "claude-sonnet-4-5-20250929",
            BASE_RUBRIC,
            {"temperature": 1.0, "max_tokens": 16},
            "same model, different temperature",
        ),
        JudgeConfigSpec(
            "e-other-provider",
            "openai",
            "gpt-4o-mini",
            BASE_RUBRIC,
            {"temperature": 0.0, "max_tokens": 16},
            "a different provider entirely",
        ),
    ]


def make_adapter(spec: JudgeConfigSpec, *, simulate: bool = False) -> JudgeAdapter:
    if simulate:
        # Exercises the whole Tier 2 pipeline without a provider key. The resulting pool
        # is stamped `simulated: true`, and `scripts/gen_results.py` refuses to present it
        # as a real-judge result — a simulated judge cannot answer a question about real
        # judges, and labelling it clearly is the only honest way to ship the code path.
        from benchlock.judge.base import SimulatedJudge

        return SimulatedJudge(
            seed=abs(hash(spec.key)) % 10_000,
            per_item_sd=0.06 if spec.key != "d-temperature" else 0.14,
            drift={"a-baseline": 0.0, "b-other-snapshot": -0.07, "c-strict-rubric": -0.12}.get(
                spec.key, 0.0
            ),
            model=f"simulated::{spec.model}",
            rubric_text=spec.rubric,
        )
    if spec.provider == "anthropic":
        from benchlock.judge.anthropic import AnthropicJudge

        return AnthropicJudge(
            model=spec.model, rubric_text=spec.rubric, scale=(1.0, 5.0), params=spec.params
        )
    if spec.provider == "openai":
        from benchlock.judge.openai import OpenAIJudge

        return OpenAIJudge(
            model=spec.model, rubric_text=spec.rubric, scale=(1.0, 5.0), params=spec.params
        )
    raise ValueError(f"unknown provider {spec.provider!r}")


def score_pool(
    items: Sequence[PoolItem],
    spec: JudgeConfigSpec,
    *,
    replicates: int = 1,
    cache: ScoreCache | None = None,
    nonce_prefix: str = "",
    simulate: bool = False,
) -> list[dict[str, float]]:
    """Score every item `replicates` times under one configuration."""
    adapter = make_adapter(spec, simulate=simulate)
    if cache is not None:
        from benchlock.judge.cache import CachedJudge

        adapter = CachedJudge(inner=adapter, cache=cache)  # type: ignore[assignment]

    scorings: list[dict[str, float]] = []
    for k in range(replicates):
        requests = [
            JudgeRequest(
                item_id=item.item_id,
                prompt_input=item.prompt_input,
                output=item.output,
                # Replicates must not be served from the provider's cache, or the noise
                # floor measures the cache rather than the judge (Phase 7.9).
                nonce=f"{nonce_prefix}rep{k}" if replicates > 1 else nonce_prefix,
            )
            for item in items
        ]
        results = adapter.score(requests)
        scorings.append({r.item_id: (r.raw_score - 1.0) / 4.0 for r in results})
    return scorings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=400)
    parser.add_argument("--replicates", type=int, default=5, help="K, on the anchor subset")
    parser.add_argument("--anchor-items", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="cost estimate only")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="build the pool with the deterministic built-in judge (no API key). The "
        "output is stamped `simulated` and is NOT a real-judge result.",
    )
    args = parser.parse_args(argv)

    items = load_items(args.items)
    configs = default_configs()
    anchor_subset = items[: args.anchor_items]

    total_calls = len(items) * len(configs) + len(anchor_subset) * (args.replicates - 1)
    estimated = (
        sum(make_adapter(spec).estimate_cost(len(items)).dollars for spec in configs)
        + make_adapter(configs[0]).estimate_cost(len(anchor_subset) * (args.replicates - 1)).dollars
    )

    print(f"pool: {len(items)} items x {len(configs)} configs")
    print(f"noise floor: {len(anchor_subset)} anchors x {args.replicates} replicates")
    print(f"total calls: ~{total_calls:,}")
    print(f"estimated spend: ${estimated:.2f}")
    if args.dry_run:
        return 0

    cache = None if args.simulate else ScoreCache(ROOT / ".benchlock" / "pool-cache.sqlite")
    started = time.time()
    pool: dict[str, dict[str, float]] = {}
    replicate_scorings: list[dict[str, float]] = []
    spend = 0.0
    calls = 0

    for spec in configs:
        print(f"  scoring under {spec.key} ({spec.why})...", flush=True)
        scorings = score_pool(items, spec, cache=cache, simulate=args.simulate)
        pool[spec.key] = scorings[0]
        calls += len(items)
        spend += 0.0 if args.simulate else make_adapter(spec).estimate_cost(len(items)).dollars

    print(f"  {args.replicates} replicates on {len(anchor_subset)} anchors...", flush=True)
    replicate_scorings = score_pool(
        anchor_subset,
        configs[0],
        replicates=args.replicates,
        cache=cache,
        simulate=args.simulate,
    )
    calls += len(anchor_subset) * args.replicates

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": "uv run python bench/real/build_pool.py"
        + (" --simulate" if args.simulate else ""),
        "simulated": bool(args.simulate),
        "dataset": items[0].dataset if items else "",
        "n_items": len(items),
        "anchor_items": len(anchor_subset),
        "replicates": args.replicates,
        "configs": [
            {"key": c.key, "provider": c.provider, "model": c.model, "why": c.why} for c in configs
        ],
        # Item ids are hashed here as they are in the ledger: the pool is committed, and a
        # dataset id can carry content.
        "scores": {
            key: {hash_item_id(item): score for item, score in scoring.items()}
            for key, scoring in pool.items()
        },
        "replicate_scores": [
            {hash_item_id(item): score for item, score in scoring.items()}
            for scoring in replicate_scorings
        ],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (RESULTS / "real-pool.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    (RESULTS / "cost.json").write_text(
        json.dumps(
            {
                "command": "uv run python bench/real/build_pool.py",
                "total_calls": calls,
                "total_dollars": round(spend, 4),
                "note": "estimated from published per-token prices and observed token counts",
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote bench/results/real-pool.json ({calls:,} calls, ~${spend:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
