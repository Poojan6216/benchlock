"""Tier 1: the simulation study.

Every method, every cell of the grid, with the full metric set. The rule this file exists
to enforce is that **misattribution and detection delay are always reported together**.
Reporting misattribution alone rewards a method that never decides anything; reporting
delay alone rewards one that fires constantly. Either number in isolation is a way of
winning a benchmark without being useful, so both appear in every table or neither does.

    uv run python bench/sim/run_sim.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Run as a script from the repo root as well as imported as a module.
if __package__ in (None, ""):  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bench.sim.baselines import (
    SINGLE_STREAM,
    Method,
    MethodResult,
    StreamView,
    all_methods,
    arl0,
)
from bench.sim.generate import StreamSpec, generate
from benchlock.config import AttributionConfig

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "bench" / "results"

#: The grid. Judge shift x system shift is the whole point: the cells where exactly one
#: moved are where attribution either works or does not.
JUDGE_SHIFTS = (0.0, 0.02, 0.05, 0.10)
SYSTEM_SHIFTS = (0.0, 0.02, 0.05, 0.10)
CHANGE_POINTS = (10, 25)
NOISE_LEVELS = (0.08, 0.16)
SCORE_TYPES = ("continuous", "likert5", "binary")


@dataclass(frozen=True, slots=True)
class Cell:
    judge_shift: float
    system_shift: float
    change_at: int
    per_item_sd: float
    score_type: str
    heteroscedastic: bool = False

    @property
    def truth(self) -> str:
        if self.judge_shift and self.system_shift:
            return "both"
        if self.judge_shift:
            return "judge"
        if self.system_shift:
            return "system"
        return "stable"

    def key(self) -> str:
        return (
            f"dj{self.judge_shift}_ds{self.system_shift}_cp{self.change_at}"
            f"_sd{self.per_item_sd}_{self.score_type}" + ("_het" if self.heteroscedastic else "")
        )


@dataclass
class Tally:
    """Counts for one (method, cell) pair."""

    n: int = 0
    alarms: int = 0
    delays: list[int] = field(default_factory=list)
    alarm_times: list[int | None] = field(default_factory=list)
    verdicts: dict[str, int] = field(default_factory=dict)

    def add(self, result: MethodResult, change_at_monitored: int | None) -> None:
        self.n += 1
        self.verdicts[result.verdict] = self.verdicts.get(result.verdict, 0) + 1
        self.alarm_times.append(result.alarm_time)
        if result.alarm_time is not None:
            self.alarms += 1
            if change_at_monitored is not None and result.alarm_time >= change_at_monitored:
                self.delays.append(result.alarm_time - change_at_monitored)


def stream_seed(*parts: object) -> int:
    """A seed that is the same on every machine, every process, forever.

    Python's builtin `hash()` is randomised per process for `str` (PEP 456), so
    `hash((cell.key(), seed))` drew a DIFFERENT stream on every invocation. Every Tier 1
    number was therefore unreproducible: the command printed under each published table
    could not regenerate the table, and two runs of the same code disagreed by Monte-Carlo
    noise that looked like a real change. Hard Rule 7 asks for determinism; this is what
    delivers it.
    """
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def build_view(spec: StreamSpec, config: AttributionConfig) -> StreamView:
    system, anchor = generate(spec)
    return StreamView(
        system_means=tuple(r.mean for r in system),
        anchor_means=tuple(r.mean for r in anchor),
        baseline_runs=config.baseline_runs,
        obs_per_run=spec.system_items,
        system_items=tuple(r.scores() for r in system),
        system_runs=tuple(system),
        anchor_runs=tuple(anchor),
    )


def cells(*, quick: bool) -> Iterator[Cell]:
    judge = JUDGE_SHIFTS[:2] if quick else JUDGE_SHIFTS
    system = SYSTEM_SHIFTS[:2] if quick else SYSTEM_SHIFTS
    change_points = CHANGE_POINTS[:1] if quick else CHANGE_POINTS
    noise = NOISE_LEVELS[:1] if quick else NOISE_LEVELS
    types = SCORE_TYPES[:1] if quick else SCORE_TYPES
    for dj in judge:
        for ds in system:
            for cp in change_points:
                for sd in noise:
                    for score_type in types:
                        yield Cell(dj, ds, cp, sd, score_type)


def misattribution(truth: str, verdicts: dict[str, int], n: int) -> dict[str, float]:
    """How often a method named the wrong cause, decomposed by direction.

    A single-stream method has only "regression" available, so on a judge-only stream it
    misattributes every time it fires. That is a structural property of monitoring one
    stream, and the results say so rather than presenting it as a deficiency.
    """
    if n == 0:
        return {"judge_as_system": 0.0, "system_as_judge": 0.0, "indeterminate": 0.0}
    judge_as_system = 0
    system_as_judge = 0
    if truth == "judge":
        judge_as_system = verdicts.get("system", 0) + verdicts.get("regression", 0)
    if truth == "system":
        system_as_judge = verdicts.get("judge", 0)
    return {
        "judge_as_system": judge_as_system / n,
        "system_as_judge": system_as_judge / n,
        "indeterminate": verdicts.get("indeterminate", 0) / n,
    }


def run(*, seeds: int, quick: bool, horizon: int, config: AttributionConfig) -> dict[str, Any]:
    methods: Sequence[Method] = all_methods(config)
    rows: list[dict[str, Any]] = []
    started = time.time()

    grid = list(cells(quick=quick))
    for index, cell in enumerate(grid):
        tallies: dict[str, Tally] = {m.name: Tally() for m in methods}
        change_at_monitored = max(0, cell.change_at - config.baseline_runs)
        for seed in range(seeds):
            spec = StreamSpec(
                name=cell.key(),
                seed=stream_seed(cell.key(), seed),
                n_runs=horizon,
                change_at=cell.change_at,
                per_item_sd=cell.per_item_sd,
                judge_shift=-cell.judge_shift,
                system_shift=-cell.system_shift,
                score_type=cell.score_type,  # type: ignore[arg-type]
                baseline_runs=config.baseline_runs,
            )
            view = build_view(spec, config)
            for method in methods:
                tallies[method.name].add(
                    method.run(view), change_at_monitored if cell.truth != "stable" else None
                )

        monitored = horizon - config.baseline_runs
        for method in methods:
            tally = tallies[method.name]
            rows.append(
                {
                    **asdict(cell),
                    "truth": cell.truth,
                    "method": method.name,
                    "single_stream": method.name in SINGLE_STREAM,
                    "n": tally.n,
                    "alarm_rate": tally.alarms / tally.n if tally.n else 0.0,
                    "false_alarm_rate": (tally.alarms / tally.n)
                    if cell.truth == "stable"
                    else None,
                    "arl0": arl0(tally.alarm_times, monitored) if cell.truth == "stable" else None,
                    "median_delay": float(np.median(tally.delays)) if tally.delays else None,
                    "mean_delay": float(np.mean(tally.delays)) if tally.delays else None,
                    "detection_rate": (
                        len(tally.delays) / tally.n if cell.truth != "stable" else None
                    ),
                    "verdicts": tally.verdicts,
                    **misattribution(cell.truth, tally.verdicts, tally.n),
                }
            )
        print(
            f"  [{index + 1}/{len(grid)}] {cell.key():<44s} ({time.time() - started:.0f}s)",
            flush=True,
        )

    return {
        "generated_by": "uv run python bench/sim/run_sim.py" + (" --quick" if quick else " --all"),
        "seeds_per_cell": seeds,
        "horizon": horizon,
        "baseline_runs": config.baseline_runs,
        "alpha": config.alpha,
        "target_shift": config.target_shift,
        "cells": len(grid),
        "elapsed_seconds": round(time.time() - started, 1),
        "rows": rows,
    }


#: The two headline cells (Phase 5.5). They get many more seeds than the grid because
#: they are the numbers that go at the top of the README, and a headline number with a
#: wide Monte-Carlo error is not a headline number.
HEADLINE_CELLS = (
    Cell(
        judge_shift=0.0, system_shift=0.0, change_at=25, per_item_sd=0.08, score_type="continuous"
    ),
    Cell(
        judge_shift=0.10, system_shift=0.0, change_at=25, per_item_sd=0.08, score_type="continuous"
    ),
)


def run_headline(*, seeds: int, horizon: int, config: AttributionConfig) -> dict[str, Any]:
    """Phase 5.5: false alarm under peeking, and misattribution under a silent judge change."""
    methods: Sequence[Method] = all_methods(config)
    started = time.time()
    out: dict[str, Any] = {
        "generated_by": (
            f"uv run python bench/sim/run_sim.py --headline --seeds {seeds} --horizon {horizon}"
        ),
        "seeds": seeds,
        "horizon": horizon,
        "monitored_runs": horizon - config.baseline_runs,
        "baseline_runs": config.baseline_runs,
        "alpha": config.alpha,
        "anchor_items": 200,
        "obs_per_run": 200,
        "per_item_sd": 0.08,
    }

    for cell in HEADLINE_CELLS:
        tallies: dict[str, Tally] = {m.name: Tally() for m in methods}
        for seed in range(seeds):
            spec = StreamSpec(
                name=cell.key(),
                seed=stream_seed(cell.key(), seed, "headline"),
                n_runs=horizon,
                change_at=cell.change_at,
                per_item_sd=cell.per_item_sd,
                judge_shift=-cell.judge_shift,
                system_shift=-cell.system_shift,
                baseline_runs=config.baseline_runs,
            )
            view = build_view(spec, config)
            for method in methods:
                tallies[method.name].add(method.run(view), None)
        monitored = horizon - config.baseline_runs
        out[cell.truth] = {
            method.name: {
                "single_stream": method.name in SINGLE_STREAM,
                "alarm_rate": tallies[method.name].alarms / seeds,
                "arl0": arl0(tallies[method.name].alarm_times, monitored),
                "verdicts": tallies[method.name].verdicts,
            }
            for method in methods
        }
        print(f"  headline cell `{cell.truth}` done ({time.time() - started:.0f}s)", flush=True)

    out["elapsed_seconds"] = round(time.time() - started, 1)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="the full grid")
    parser.add_argument("--quick", action="store_true", help="a small grid, for iteration")
    parser.add_argument(
        "--headline", action="store_true", help="only the two Phase 5.5 headline cells"
    )
    parser.add_argument("--seeds", type=int, default=200, help="streams per cell")
    parser.add_argument("--horizon", type=int, default=60, help="runs per stream")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.all and not args.quick and not args.headline:
        parser.error("pass --all, --quick or --headline")

    config = AttributionConfig(
        alpha=0.05, min_runs=8, min_obs=30, baseline_runs=8, target_shift=0.05
    )
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.headline:
        print(f"headline study: {args.seeds} seeds, horizon {args.horizon}")
        headline = run_headline(seeds=args.seeds, horizon=args.horizon, config=config)
        target = args.out or RESULTS / "sim-headline.json"
        target.write_text(json.dumps(headline, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {target}  ({headline['elapsed_seconds']}s)")
        return 0

    print(f"simulation study: {args.seeds} seeds/cell, horizon {args.horizon}")
    payload = run(seeds=args.seeds, quick=args.quick, horizon=args.horizon, config=config)
    out = args.out or RESULTS / "sim-latest.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"\nwrote {shown}  ({payload['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
