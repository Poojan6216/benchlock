"""Tier 2 step 3: run every method against every scenario, and report.

uv run python bench/real/run_real.py --all
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from bench.real.compose import METHODOLOGY, compose, load_pool, scenarios
from bench.sim.baselines import SINGLE_STREAM, StreamView, all_methods
from benchlock.config import AttributionConfig

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "bench" / "results"
CONFIG = AttributionConfig(alpha=0.05, min_runs=8, min_obs=30, baseline_runs=8, target_shift=0.05)

#: A single-stream method can only ever say "regression". On a judge-only scenario that is
#: wrong; on a system-only one it is right. Scored on that basis rather than pretending
#: they have verdicts they do not have.
CORRECT: dict[str, set[str]] = {
    "judge": {"judge"},
    "system": {"system", "regression"},
    "both": {"both"},
    "stable": {"stable"},
}


def judge_nondeterminism(pool: dict) -> dict:
    """Phase 6.5: how much does a temperature-0 hosted judge disagree with itself?"""
    replicate_scorings = [dict(s) for s in pool["replicate_scores"]]
    items = sorted(set.intersection(*(set(s) for s in replicate_scorings)))
    pairwise: list[float] = []
    agreements = 0
    comparisons = 0
    per_item_sds: list[float] = []
    for item in items:
        scores = [s[item] for s in replicate_scorings]
        per_item_sds.append(float(np.std(scores, ddof=1)))
        for a in range(len(scores)):
            for b in range(a + 1, len(scores)):
                diff = abs(scores[a] - scores[b])
                pairwise.append(diff)
                comparisons += 1
                agreements += int(diff == 0.0)
    run_means = [float(np.mean([s[i] for i in items])) for s in replicate_scorings]
    baseline = next(c for c in pool["configs"] if c["key"] == "a-baseline")

    # How much of the run-to-run movement is SHARED across items rather than independent?
    # Independent per-item noise would give run_mean_sd = per_item_sd / sqrt(n). Anything
    # above that is a run-level offset moving every item together — the component that no
    # anchor size can buy down (see stats/power.py::decompose).
    per_item_sd = float(np.mean(per_item_sds))
    run_mean_sd = float(np.std(run_means, ddof=1))
    predicted = per_item_sd / np.sqrt(len(items)) if items else 0.0
    unstable = sum(1 for i in items if len({s[i] for s in replicate_scorings}) > 1)

    return {
        "command": "uv run python bench/real/run_real.py --all",
        "replicates": len(replicate_scorings),
        "n_items": len(items),
        "rows": [
            {
                "provider": baseline["provider"],
                "model": baseline["model"],
                "rubric": "base 1-5 helpfulness",
                "score_type": "likert5",
                "exact_agreement_rate": agreements / comparisons if comparisons else 0.0,
                "self_disagreement_rate": 1.0 - (agreements / comparisons if comparisons else 0.0),
                "mean_abs_pairwise_diff": float(np.mean(pairwise)) if pairwise else 0.0,
                "per_item_sd": per_item_sd,
                "run_mean_sd": run_mean_sd,
                "items_that_ever_varied": unstable,
                "items_that_ever_varied_rate": unstable / len(items) if items else 0.0,
                "pairwise_comparisons": comparisons,
                # >1 means the judge's noise is correlated across items, so a bigger anchor
                # set buys less than 1/sqrt(n) would suggest.
                "independence_ratio": (run_mean_sd / predicted) if predicted > 0 else None,
                "run_means": [round(m, 5) for m in run_means],
            }
        ],
        "note": (
            "Measured from K identical calls per item, each with a per-call cache-busting "
            "nonce so the provider cannot serve a cached answer. Every number here is a "
            "property of a real hosted judge at its most deterministic setting."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.all:
        parser.error("pass --all")

    pool = load_pool()
    methods = all_methods(CONFIG)
    started = time.time()
    rows = []

    for scenario in scenarios():
        print(f"  {scenario.name} (truth: {scenario.truth})...", flush=True)
        tallies: dict[str, list[str]] = {m.name: [] for m in methods}
        for seed in range(args.seeds):
            system, anchor = compose(scenario, pool, n_runs=args.runs, seed=seed)
            view = StreamView(
                system_means=tuple(r.mean for r in system),
                anchor_means=tuple(r.mean for r in anchor),
                baseline_runs=CONFIG.baseline_runs,
                obs_per_run=system[0].n,
                system_items=tuple(r.scores() for r in system),
                system_runs=tuple(system),
                anchor_runs=tuple(anchor),
            )
            for method in methods:
                tallies[method.name].append(method.run(view).verdict)

        for method in methods:
            verdicts = tallies[method.name]
            modal = max(set(verdicts), key=verdicts.count)
            rows.append(
                {
                    "scenario": scenario.name,
                    "truth": scenario.truth,
                    "description": scenario.description,
                    "method": method.name,
                    "single_stream": method.name in SINGLE_STREAM,
                    "verdict": modal,
                    "correct": modal in CORRECT[scenario.truth],
                    "accuracy": sum(1 for v in verdicts if v in CORRECT[scenario.truth])
                    / len(verdicts),
                }
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "real-latest.json").write_text(
        json.dumps(
            {
                "command": "uv run python bench/real/run_real.py --all",
                "methodology": METHODOLOGY,
                "runs_per_stream": args.runs,
                "seeds_per_scenario": args.seeds,
                "elapsed_seconds": round(time.time() - started, 1),
                "rows": rows,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    (RESULTS / "judge-nondeterminism.json").write_text(
        json.dumps(judge_nondeterminism(pool), indent=1, sort_keys=True) + "\n"
    )
    print(f"\nwrote bench/results/real-latest.json ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
