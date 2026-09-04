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
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
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


def default_configs(*, include_openai: bool = False) -> list[JudgeConfigSpec]:
    """The judge configurations the pool is scored under.

    Model ids are the current, undated strings — a date suffix that used to be valid is
    now simply a 404, and a stale price makes `benchlock plan` under-quote. Sonnet-tier is
    the baseline because that is what teams actually run an LLM judge on.

    Note what config (d) is **not**: the spec asks for "same model at a different
    temperature", but current models reject `temperature` with a 400 — sampling parameters
    were replaced by thinking and `output_config.effort`. Effort is the honest modern
    analogue of that knob, and it is the one a real team would change.
    """
    # Request parameters are per-model, because the API surface genuinely differs.
    # Sonnet 5 takes thinking + `output_config.effort` and REJECTS temperature; Haiku 4.5
    # predates effort and rejects it with a 400, defaulting to no thinking when the
    # `thinking` key is simply omitted. Sending one shape to both fails half the run.
    sonnet_params = {
        "max_tokens": 16,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
    }
    haiku_params = {"max_tokens": 16}
    configs = [
        JudgeConfigSpec(
            "a-baseline",
            "anthropic",
            "claude-sonnet-5",
            BASE_RUBRIC,
            dict(sonnet_params),
            "the baseline judge",
        ),
        JudgeConfigSpec(
            "b-other-snapshot",
            "anthropic",
            "claude-haiku-4-5",
            BASE_RUBRIC,
            dict(haiku_params),
            "a different model from the same provider — the snapshot-rotation analogue",
        ),
        JudgeConfigSpec(
            "c-strict-rubric",
            "anthropic",
            "claude-sonnet-5",
            STRICT_RUBRIC,
            dict(sonnet_params),
            "same model, stricter rubric",
        ),
        JudgeConfigSpec(
            "d-effort",
            "anthropic",
            "claude-sonnet-5",
            BASE_RUBRIC,
            {**sonnet_params, "output_config": {"effort": "high"}},
            "same model and rubric, more reasoning effort",
        ),
    ]
    if include_openai:
        configs.append(
            JudgeConfigSpec(
                "e-other-provider",
                "openai",
                "gpt-4o-mini",
                BASE_RUBRIC,
                {"temperature": 0.0, "max_tokens": 16},
                "a different provider entirely",
            )
        )
    return configs


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


def _score_concurrent(
    spec: JudgeConfigSpec,
    requests: list[JudgeRequest],
    *,
    simulate: bool = False,
    workers: int = 8,
    label: str = "",
) -> dict[str, float]:
    """Score many items concurrently, tolerating individual failures but never hiding them.

    2,600 sequential HTTPS round-trips is over an hour of wall clock for work that is
    entirely IO-bound. Each worker builds its **own** adapter from the spec: adapters keep
    mutable token counters, and sharing one across threads would corrupt the cost
    accounting Hard Rule 5 depends on.

    Two things this function is careful about, both learned the hard way:

    * ``future.result()`` is always called. An earlier version iterated ``as_completed``
      without it, so an exception inside a worker vanished silently — the run reported
      "10/10 complete" and wrote an empty pool. A benchmark that fails quietly is worse
      than one that crashes.
    * Adapters are constructed from the **spec**, not by reflecting on an adapter instance,
      which breaks the moment the adapter is wrapped (by the cache, say).

    A call that fails after the SDK's own retries is recorded and skipped: losing one item
    of 400 is a rounding error; losing 2,000 paid calls to one bad item is not.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    reference = make_adapter(spec, simulate=simulate)
    description = reference.describe()
    lo, hi = description.scale

    scores: dict[str, float] = {}
    failures: list[tuple[str, str]] = []
    tokens = [0, 0]
    lock = Lock()
    local = threading.local()

    def score_one(request: JudgeRequest) -> None:
        try:
            if not hasattr(local, "adapter"):
                local.adapter = make_adapter(spec, simulate=simulate)
            result = local.adapter.score([request])[0]
        except Exception as exc:
            with lock:
                failures.append((request.item_id, f"{type(exc).__name__}: {exc}"[:160]))
            return
        with lock:
            scores[request.item_id] = (result.raw_score - lo) / (hi - lo)
            tokens[0] += result.input_tokens
            tokens[1] += result.output_tokens

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score_one, r) for r in requests]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()  # re-raise anything score_one did not catch itself
            if done % 100 == 0 or done == len(futures):
                print(f"      {label} {done}/{len(futures)}", flush=True)

    SPEND.add(description, tokens[0], tokens[1], len(requests) - len(failures))
    if failures:
        print(
            f"      ! {len(failures)}/{len(requests)} call(s) failed — e.g. {failures[0][1]}",
            flush=True,
        )
        FAILURES.extend(failures)
    if not scores:
        raise SystemExit(
            f"every call in `{spec.key}` failed — refusing to continue and spend more.\n"
            f"  first failure: {failures[0][1] if failures else 'unknown'}"
        )
    return scores


class _Spend:
    """Running total of what the study actually cost, from observed token counts."""

    def __init__(self) -> None:
        self.dollars = 0.0
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = Lock()

    def add(self, description: Any, input_tokens: int, output_tokens: int, calls: int) -> None:
        with self._lock:
            self.dollars += (
                input_tokens / 1e6 * description.input_cost_per_mtok
                + output_tokens / 1e6 * description.output_cost_per_mtok
            )
            self.calls += calls
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens


SPEND = _Spend()
FAILURES: list[tuple[str, str]] = []


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
        label = f"rep {k + 1}/{replicates}" if replicates > 1 else spec.key
        scorings.append(_score_concurrent(spec, requests, simulate=simulate, label=label))
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
    configs = default_configs(include_openai=bool(os.environ.get("OPENAI_API_KEY")))
    if len(configs) == 4:
        print("no OPENAI_API_KEY — running 4 Anthropic configs, skipping cross-provider")
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

    print(f"  {args.replicates} replicates on {len(anchor_subset)} anchors...", flush=True)
    replicate_scorings = score_pool(
        anchor_subset,
        configs[0],
        replicates=args.replicates,
        cache=cache,
        simulate=args.simulate,
    )
    calls += len(anchor_subset) * args.replicates
    # Measured from observed token counts, not the pre-run estimate (Hard Rule 5).
    if not args.simulate:
        calls, spend = SPEND.calls, SPEND.dollars

    # An item a safety classifier declined is scored by some configs and not others.
    # Comparing configs across different item sets would attribute a composition
    # difference to the judge, so the pool is intersected down to what all of them scored.
    common = set.intersection(*(set(v) for v in pool.values())) if pool else set()
    dropped = {k: len(v) - len(common) for k, v in pool.items() if len(v) != len(common)}
    if dropped:
        print(f"  intersected pool to {len(common)} items scored by every config {dropped}")
    pool = {k: {i: v[i] for i in common} for k, v in pool.items()}
    replicate_common = (
        set.intersection(*(set(r) for r in replicate_scorings)) if replicate_scorings else set()
    )
    replicate_scorings = [{i: r[i] for i in replicate_common} for r in replicate_scorings]

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
                "input_tokens": SPEND.input_tokens,
                "output_tokens": SPEND.output_tokens,
                "failed_calls": len(FAILURES),
                "note": (
                    "measured from the token counts the API actually reported, priced at "
                    "published per-token rates — not a pre-run estimate"
                ),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote bench/results/real-pool.json ({calls:,} calls, ~${spend:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
