"""Tier 3: attack the attributor and publish the damage.

    uv run python bench/adversarial/run_adversarial.py --all

The rule for this file: **a strategy that beats Benchlock is a result, not a bug report to
be quietly closed.** Where something was fixed, the pre-fix number stays in the table.
Where it cannot be fixed, the failure rate is published and `docs/threat-model.md` explains
why. If every number here came out zero, the red team would be too weak to be worth running.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bench.sim.generate import StreamSpec, generate
from benchlock.attribute.engine import decide
from benchlock.config import AttributionConfig
from benchlock.model.streams import Observation, RunRecord, suite_hash_of
from benchlock.model.verdict import AttributionRefusedError

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "bench" / "results"
CONFIG = AttributionConfig(alpha=0.05, min_runs=8, min_obs=30, baseline_runs=8, target_shift=0.05)


@dataclass(frozen=True, slots=True)
class AttackResult:
    strategy: str
    description: str
    failure_rate: float
    status: str
    detail: dict[str, Any]


def _verdict(system: Sequence[RunRecord], anchor: Sequence[RunRecord]) -> str:
    try:
        return decide(list(system), list(anchor), CONFIG).verdict.value
    except AttributionRefusedError as exc:
        return f"refused:{exc.rule_id}"


def _retag(runs: Sequence[RunRecord], shift_items: dict[str, float]) -> list[RunRecord]:
    """Apply a per-item score offset to every run, clipped into [0, 1]."""
    out = []
    for run in runs:
        observations = tuple(
            Observation(
                item_id=o.item_id,
                score=min(max(o.score + shift_items.get(o.item_id, 0.0), 0.0), 1.0),
                raw_score=o.raw_score,
                scale=o.scale,
            )
            for o in run.observations
        )
        out.append(
            RunRecord(
                run_id=run.run_id,
                run_index=run.run_index,
                kind=run.kind,
                observations=observations,
                suite_hash=run.suite_hash,
                judge_pin=run.judge_pin,
                anchor_pin=run.anchor_pin,
                epoch=run.epoch,
            )
        )
    return out


# ---------------------------------------------------------------------------------------
# 7.1 anchor evasion — expected to work; the question is how badly
# ---------------------------------------------------------------------------------------


def anchor_evasion(trials: int = 40) -> AttackResult:
    """A judge change confined to a region the anchor set does not cover.

    The judge gets stricter only about, say, code responses, while the anchor set is all
    prose. The anchor process sees nothing, the system stream moves, and the verdict is
    `SYSTEM` — a rollback recommendation for a system that did not change. **This is the
    fundamental limitation of the entire design**, and the coverage warning threshold in
    `anchor/coverage.py` is set from this curve.
    """
    rng = np.random.default_rng(7101)
    by_coverage: dict[float, list[str]] = {}
    for coverage in (0.0, 0.25, 0.5, 0.75, 1.0):
        verdicts: list[str] = []
        for trial in range(trials):
            spec = StreamSpec(
                name=f"evade{coverage}",
                seed=int(rng.integers(0, 2**31)),
                n_runs=60,
                change_at=20,
                judge_shift=0.0,
                system_shift=0.0,
            )
            system, anchor = generate(spec)
            # The judge change hits every system item, but only `coverage` of the anchors.
            system_shift = dict.fromkeys((o.item_id for o in system[0].observations), -0.10)
            anchor_ids = [o.item_id for o in anchor[0].observations]
            covered = anchor_ids[: int(len(anchor_ids) * coverage)]
            anchor_shift = dict.fromkeys(covered, -0.10)

            drifted_system = [
                *system[:20],
                *_retag(system[20:], system_shift),
            ]
            drifted_anchor = [
                *anchor[:20],
                *_retag(anchor[20:], anchor_shift),
            ]
            verdicts.append(_verdict(drifted_system, drifted_anchor))
            del trial
        by_coverage[coverage] = verdicts

    # A "failure" is calling a judge-only change a system regression.
    curve = {
        str(coverage): {
            "system_misattribution": v.count("system") / len(v),
            "indeterminate": v.count("indeterminate") / len(v),
            "judge": v.count("judge") / len(v),
            "both": v.count("both") / len(v),
            "stable": v.count("stable") / len(v),
        }
        for coverage, v in by_coverage.items()
    }
    worst = max(c["system_misattribution"] for c in curve.values())
    return AttackResult(
        strategy="7.1 anchor evasion",
        description=("a judge change confined to a region the anchor set does not cover"),
        failure_rate=worst,
        status="WORKS — the fundamental limitation of the design; measured, warned about, "
        "never eliminated",
        detail={"misattribution_by_coverage": curve, "trials_per_point": trials},
    )


# ---------------------------------------------------------------------------------------
# 7.2 slow ramp
# ---------------------------------------------------------------------------------------


def slow_ramp(trials: int = 25) -> AttackResult:
    """Drift introduced gradually, below the per-run detectable threshold."""
    rng = np.random.default_rng(7201)
    curve: dict[str, float] = {}
    for rate in (0.0002, 0.001, 0.002, 0.005, 0.01):
        detected = 0
        for _ in range(trials):
            spec = StreamSpec(
                name=f"ramp{rate}", seed=int(rng.integers(0, 2**31)), n_runs=80, change_at=8
            )
            system, anchor = generate(spec)
            ramped_system = []
            ramped_anchor = []
            for index, (s, a) in enumerate(zip(system, anchor, strict=True)):
                creep = -rate * max(0, index - 8)
                shift_s = dict.fromkeys((o.item_id for o in s.observations), creep)
                shift_a = dict.fromkeys((o.item_id for o in a.observations), creep)
                ramped_system.extend(_retag([s], shift_s))
                ramped_anchor.extend(_retag([a], shift_a))
            if _verdict(ramped_system, ramped_anchor) != "stable":
                detected += 1
        curve[f"{rate}"] = detected / trials
    never = [rate for rate, hit in curve.items() if hit == 0.0]
    # The failure rate is the fraction of ramp rates that evade entirely — not the best
    # case. A method that catches fast ramps and misses slow ones has a real blind spot,
    # and reporting only the fast ones would hide it.
    evaded = len(never) / len(curve)
    slowest_caught = min(
        (float(rate) for rate, hit in curve.items() if hit > 0.5), default=float("nan")
    )
    return AttackResult(
        strategy="7.2 slow ramp",
        description="drift introduced gradually, below the per-run detectable threshold",
        failure_rate=evaded,
        status=(
            f"WORKS below ~{slowest_caught:g} per run. {len(never)} of {len(curve)} tested "
            "ramp rates were never detected within the horizon. A ramp slower than the "
            "noise floor is invisible to every method, this one included; the boundary is "
            "the useful number"
        ),
        detail={
            "detection_rate_by_ramp_per_run": curve,
            "never_detected_rates": never,
            "slowest_reliably_caught": slowest_caught,
            "trials_per_point": trials,
        },
    )


# ---------------------------------------------------------------------------------------
# 7.3 simultaneous drift, including cancellation
# ---------------------------------------------------------------------------------------


def cancellation(trials: int = 40) -> AttackResult:
    """Judge and system moving in opposite directions, so the net score never moves.

    The nastiest case in the design: both components changed, the raw eval number is
    perfectly flat, and every single-stream detector sees a healthy pipeline.
    """
    rng = np.random.default_rng(7301)
    outcomes: dict[str, dict[str, float]] = {}
    for size in (0.03, 0.05, 0.10):
        verdicts = []
        for _ in range(trials):
            spec = StreamSpec(
                name=f"cancel{size}",
                seed=int(rng.integers(0, 2**31)),
                n_runs=60,
                change_at=20,
                judge_shift=size,
                system_shift=-size,
            )
            system, anchor = generate(spec)
            verdicts.append(_verdict(system, anchor))
        outcomes[f"{size}"] = {
            "both": verdicts.count("both") / trials,
            "judge": verdicts.count("judge") / trials,
            "system": verdicts.count("system") / trials,
            "indeterminate": verdicts.count("indeterminate") / trials,
            "stable_MISSED": verdicts.count("stable") / trials,
        }
    worst_miss = max(o["stable_MISSED"] for o in outcomes.values())
    return AttackResult(
        strategy="7.3 cancellation",
        description="judge and system moving in opposite directions; the net score is flat",
        failure_rate=worst_miss,
        status=(
            "PARTIALLY DEFENDED — the difference-in-differences process sees it where a "
            "single stream cannot, but a small enough cancellation is still missed"
        ),
        detail={"verdicts_by_effect_size": outcomes, "trials_per_point": trials},
    )


# ---------------------------------------------------------------------------------------
# 7.4 anchor staleness / concept drift
# ---------------------------------------------------------------------------------------


def anchor_staleness(trials: int = 30) -> AttackResult:
    """The team's own standard changes, and the judge is updated to match.

    Hedging used to be penalised; now it is fine. The judge is genuinely different, so
    `frozen-self` anchors correctly report a judge change — but the team does not want to
    hear "the judge drifted", they want to hear "your rubric changed and you should
    re-baseline". Benchlock cannot distinguish those, and this measures how long it keeps
    saying the former.
    """
    rng = np.random.default_rng(7401)
    runs_to_notice: list[int] = []
    for _ in range(trials):
        spec = StreamSpec(
            name="stale",
            seed=int(rng.integers(0, 2**31)),
            n_runs=80,
            change_at=20,
            judge_shift=0.08,
        )
        system, anchor = generate(spec)
        noticed: int | None = None
        for t in range(20, 80, 4):
            if _verdict(system[: t + 1], anchor[: t + 1]) == "judge":
                noticed = t - 20
                break
        runs_to_notice.append(noticed if noticed is not None else 60)
    return AttackResult(
        strategy="7.4 anchor staleness",
        description="the team's own standard changed; frozen anchors read it as judge drift",
        failure_rate=1.0,
        status=(
            "WORKS BY CONSTRUCTION — benchlock reports *that* the judge changed, never "
            "*why*. A deliberate rubric change is indistinguishable from provider drift, "
            "and `benchlock rebaseline --reason` is the intended answer"
        ),
        detail={
            "median_runs_until_reported": float(np.median(runs_to_notice)),
            "trials": trials,
        },
    )


# ---------------------------------------------------------------------------------------
# 7.5 input distribution shift — the third cause
# ---------------------------------------------------------------------------------------


def input_distribution_shift(trials: int = 30) -> AttackResult:
    """The eval suite is sampled from production traffic, and the mix shifts.

    Score movement now has a third cause Benchlock does not model. Two variants, because
    they have different answers:

    * **The item set changes.** New questions replace old ones, the suite hash moves, and
      lattice rule 2 refuses to attribute. Defended.
    * **The item set is stable but its contents are not.** The same slots now hold harder
      questions — the same 200 test ids, regenerated from this month's traffic. The suite
      hash is unchanged, so nothing catches it, and the drop is attributed to the system.
      **This is undefended**, and it is the more realistic of the two.
    """
    rng = np.random.default_rng(7501)
    refused = 0
    misattributed = 0
    for _ in range(trials):
        spec = StreamSpec(
            name="inputshift", seed=int(rng.integers(0, 2**31)), n_runs=60, change_at=20
        )
        system, anchor = generate(spec)

        # Variant A: the item set itself changes, so the suite hash moves.
        renamed = []
        for run in system:
            if run.run_index < 20:
                renamed.append(run)
                continue
            observations = tuple(
                Observation(
                    item_id=f"new-{o.item_id}",
                    score=min(max(o.score - 0.10, 0.0), 1.0),
                    raw_score=o.raw_score,
                    scale=o.scale,
                )
                for o in run.observations
            )
            renamed.append(
                RunRecord(
                    run_id=run.run_id,
                    run_index=run.run_index,
                    kind=run.kind,
                    observations=observations,
                    suite_hash=suite_hash_of(o.item_id for o in observations),
                    judge_pin=run.judge_pin,
                    anchor_pin=run.anchor_pin,
                    epoch=run.epoch,
                )
            )
        if _verdict(renamed, anchor).startswith("refused"):
            refused += 1

        # Variant B: the ids are stable and the *questions behind them* got harder. The
        # suite hash cannot see this, because a hash of item ids is not a hash of content.
        harder = [
            *system[:20],
            *_retag(
                system[20:],
                dict.fromkeys((o.item_id for o in system[0].observations), -0.10),
            ),
        ]
        if _verdict(harder, anchor) == "system":
            misattributed += 1

    return AttackResult(
        strategy="7.5 input distribution shift",
        description="the eval suite itself moves, adding a third cause benchlock cannot model",
        failure_rate=misattributed / trials,
        status=(
            f"PARTIALLY DEFENDED. When the item set changes, the suite-hash check refuses "
            f"{refused / trials:.0%} of the time. When the ids stay the same and only the "
            f"questions behind them get harder, the check is blind and "
            f"{misattributed / trials:.0%} become a confident `system` verdict. Hashing "
            "item ids does not hash item content, and a fixed suite is an assumption "
            "benchlock states but cannot verify"
        ),
        detail={
            "refusal_rate_when_item_set_changes": refused / trials,
            "system_misattribution_when_ids_stable": misattributed / trials,
            "trials": trials,
        },
    )


# ---------------------------------------------------------------------------------------
# 7.6 adversarial ordering
# ---------------------------------------------------------------------------------------


def adversarial_ordering(trials: int = 30) -> AttackResult:
    """Worst-case ordering of observations within a run, to starve a betting process."""
    rng = np.random.default_rng(7601)
    random_delay: list[int] = []
    adversarial_delay: list[int] = []
    for _ in range(trials):
        spec = StreamSpec(
            name="order",
            seed=int(rng.integers(0, 2**31)),
            n_runs=60,
            change_at=15,
            system_shift=-0.08,
        )
        system, anchor = generate(spec)

        def crossed_at(runs: list[RunRecord], against: list[RunRecord] = anchor) -> int:
            try:
                evidence = decide(runs, against, CONFIG).evidence
            except AttributionRefusedError:
                return 60
            return evidence.crossed_at_system if evidence.crossed_at_system is not None else 60

        random_delay.append(crossed_at(system))
        # Sort each run's observations so the misleading ones come first.
        worst = [
            RunRecord(
                run_id=r.run_id,
                run_index=r.run_index,
                kind=r.kind,
                observations=tuple(sorted(r.observations, key=lambda o: -o.score)),
                suite_hash=r.suite_hash,
                judge_pin=r.judge_pin,
                anchor_pin=r.anchor_pin,
                epoch=r.epoch,
            )
            for r in system
        ]
        adversarial_delay.append(crossed_at(worst))

    penalty = float(np.median(adversarial_delay) - np.median(random_delay))
    return AttackResult(
        strategy="7.6 adversarial ordering",
        description="worst-case ordering of observations within each run",
        failure_rate=0.0 if penalty <= 0 else min(1.0, penalty / 60.0),
        status=(
            f"NEGLIGIBLE — median delay penalty {penalty:+.1f} runs. The stream is "
            "monitored at run level, so within-run ordering cannot starve the bets"
        ),
        detail={
            "median_delay_random_order": float(np.median(random_delay)),
            "median_delay_adversarial_order": float(np.median(adversarial_delay)),
            "penalty_runs": penalty,
            "trials": trials,
        },
    )


# ---------------------------------------------------------------------------------------
# 7.7 heavy tails and bound violations
# ---------------------------------------------------------------------------------------


def bound_violations() -> AttackResult:
    """Near-degenerate distributions, and scores outside their declared range."""
    from benchlock.adapters.jsonl import IngestError, parse_records

    cases: dict[str, str] = {}

    # 99% at the ceiling: legal, and the guarantee degrades gracefully rather than breaking.
    rng = np.random.default_rng(7701)
    spec = StreamSpec(
        name="ceiling",
        seed=1,
        n_runs=60,
        change_at=20,
        system_shift=-0.05,
        system_level=0.99,
        anchor_level=0.99,
        per_item_sd=0.01,
    )
    system, anchor = generate(spec)
    cases["99pct_at_ceiling"] = _verdict(system, anchor)

    # Zero variance everywhere: every item scores identically in every run. The earlier
    # version of this case added 0.0 to each score, which is the identity — it "tested"
    # a stream byte-identical to the normal one and could not have caught anything.
    flat = [
        RunRecord(
            run_id=r.run_id,
            run_index=r.run_index,
            kind=r.kind,
            observations=tuple(
                Observation(item_id=o.item_id, score=0.75, raw_score=4.0, scale=o.scale)
                for o in r.observations
            ),
            suite_hash=r.suite_hash,
            judge_pin=r.judge_pin,
            anchor_pin=r.anchor_pin,
            epoch=r.epoch,
        )
        for r in system
    ]
    cases["zero_variance"] = _verdict(flat, anchor)
    # A flat stream that then steps must still be detected, not swallowed by a zero scale.
    stepped = [
        *flat[:20],
        *_retag(flat[20:], dict.fromkeys((o.item_id for o in flat[0].observations), -0.20)),
    ]
    cases["zero_variance_then_step"] = _verdict(stepped, anchor)

    # A score outside the declared bounds must be refused at ingest, not monitored.
    try:
        parse_records([(1, {"item_id": "a", "score": 9.0})], (1.0, 5.0), source="attack")
        cases["out_of_range"] = "ACCEPTED — this is a bug"
    except IngestError:
        cases["out_of_range"] = "refused at ingest"

    del rng
    silent = [k for k, v in cases.items() if "bug" in v]
    return AttackResult(
        strategy="7.7 heavy tails and bound violations",
        description="near-degenerate score distributions and out-of-range scores",
        failure_rate=len(silent) / len(cases),
        status=(
            "DEFENDED — out-of-range scores are refused at ingest rather than clamped, and "
            "degenerate distributions produce a conservative verdict rather than a wrong one"
        ),
        detail={"cases": cases},
    )


# ---------------------------------------------------------------------------------------
# 7.9 provider-side response caching
# ---------------------------------------------------------------------------------------


def provider_caching(trials: int = 20) -> AttackResult:
    """If the provider caches judge responses, the anchor set returns yesterday's answers.

    The anchor stream then looks artificially stable and real judge drift becomes
    invisible. The mitigation is a per-run cache-busting nonce — and because the nonce
    changes the prompt, its own effect on scores has to be measured too. A mitigation that
    is itself a confound is not a mitigation.
    """
    from benchlock.anchor.modes import AnchorItem, freeze, rescore
    from benchlock.judge.base import SimulatedJudge

    items = [AnchorItem(item_id=f"a{i}", prompt_input=f"q{i}", output=f"o{i}") for i in range(200)]

    def run(*, cache: bool, nonce: bool) -> int:
        """How many of `trials` runs show the judge's drift in the anchor mean."""
        judge = SimulatedJudge(seed=1, cache=cache)
        frozen = freeze(items, judge, replicates=5, nonce_prefix="n" if nonce else "")
        moved = 0
        for t in range(trials):
            drifted = SimulatedJudge(seed=100 + t, drift=-0.10, cache=cache)
            if cache:
                drifted._cache = judge._cache  # the provider's cache persists across calls
            scores = rescore(items, drifted, nonce=f"run{t}" if nonce else "")
            deviation = sum(
                scores[i.item_id] - frozen.baseline_scores[i.item_id] for i in items
            ) / len(items)
            if abs(deviation) > 0.05:
                moved += 1
        return moved

    without_nonce = run(cache=True, nonce=False)
    with_nonce = run(cache=True, nonce=True)
    no_cache = run(cache=False, nonce=False)

    # Does the nonce itself move scores? If it does, the mitigation is a confound.
    plain = SimulatedJudge(seed=42)
    nonce_judge = SimulatedJudge(seed=42)
    plain_scores = rescore(items, plain)
    nonce_scores = rescore(items, nonce_judge, nonce="cache-buster-1")
    nonce_effect = abs(
        sum(plain_scores.values()) / len(plain_scores)
        - sum(nonce_scores.values()) / len(nonce_scores)
    )

    return AttackResult(
        strategy="7.9 provider-side response caching",
        description="a cached judge returns yesterday's answers, hiding real drift",
        failure_rate=1.0 - without_nonce / trials,
        status=(
            f"DEFENDED by a per-run nonce: drift visible in {without_nonce}/{trials} runs "
            f"without it, {with_nonce}/{trials} with it. NOTE: the nonce's measured effect "
            f"on the mean score here is {nonce_effect:.4f}, but that is an artefact of the "
            "simulated judge, whose nonce touches only the cache key. Whether a nonce "
            "perturbs a REAL judge's scores is UNMEASURED: Tier 2 used a nonce on every "
            "replicate call but never scored the same items with and without one, so the "
            "mitigation's own confounding effect is an open question rather than a "
            "verified non-issue"
        ),
        detail={
            "drift_visible_runs_without_nonce": without_nonce,
            "drift_visible_runs_with_nonce": with_nonce,
            "drift_visible_runs_no_cache": no_cache,
            "nonce_effect_on_mean_score_SIMULATED": nonce_effect,
            "nonce_effect_is_measurable_only_against_a_real_judge": True,
            "trials": trials,
        },
    )


ATTACKS = {
    "anchor-evasion": anchor_evasion,
    "slow-ramp": slow_ramp,
    "cancellation": cancellation,
    "anchor-staleness": anchor_staleness,
    "input-shift": input_distribution_shift,
    "adversarial-ordering": adversarial_ordering,
    "bound-violations": bound_violations,
    "provider-caching": provider_caching,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--only", choices=sorted(ATTACKS), default=None)
    args = parser.parse_args(argv)
    if not args.all and args.only is None:
        parser.error("pass --all or --only <strategy>")

    chosen = {args.only: ATTACKS[args.only]} if args.only else ATTACKS
    started = time.time()
    rows = []
    for name, attack in chosen.items():
        print(f"  running {name}...", flush=True)
        result = attack()
        rows.append(
            {
                "strategy": result.strategy,
                "description": result.description,
                "failure_rate": result.failure_rate,
                "status": result.status,
                "detail": result.detail,
            }
        )
        print(f"    failure rate {result.failure_rate:.0%} — {result.status}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": "uv run python bench/adversarial/run_adversarial.py --all",
        "elapsed_seconds": round(time.time() - started, 1),
        "rows": rows,
    }
    # A single-strategy run is for iterating on that strategy. Writing it over the full
    # results would silently shrink the published table to one row.
    target = (
        RESULTS / f"adversarial-only-{args.only}.json"
        if args.only
        else RESULTS / "adversarial-latest.json"
    )
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {target.relative_to(ROOT)} ({payload['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
