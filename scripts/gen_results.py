#!/usr/bin/env python3
"""Generate RESULTS.md from committed measurement JSON.

`RESULTS.md` is generated. It is never hand-edited (Hard Rule 5). Every number in it
traces to a file in `bench/results/` and the command that produced that file.

    uv run python scripts/gen_results.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "RESULTS.md"

METHOD_LABELS = {
    "B0-fixed-threshold": "B0 fixed threshold",
    "B1-peeking-t-test": "B1 peeking t-test",
    "B2-bonferroni-t-test": "B2 Bonferroni t-test",
    "B3-cusum": "B3 CUSUM",
    "B4-adwin": "B4 ADWIN",
    "B4-ddm": "B4 DDM",
    "B5-benchlock-no-anchor": "B5 Benchlock, no anchor",
    "B6-benchlock": "B6 **Benchlock**",
}
ORDER = list(METHOD_LABELS)


def load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value: float | None, spec: str = ".3f") -> str:
    return "—" if value is None else format(value, spec)


def headline_section(headline: dict[str, Any]) -> str:
    stable, judge = headline["stable"], headline["judge"]
    rows = []
    for name in ORDER:
        if name not in stable:
            continue
        verdicts = judge[name]["verdicts"]
        total = sum(verdicts.values()) or 1
        wrong = (verdicts.get("regression", 0) + verdicts.get("system", 0)) / total
        right = verdicts.get("judge", 0) / total
        unsure = verdicts.get("indeterminate", 0) / total
        missed = verdicts.get("stable", 0) / total
        rows.append(
            f"| {METHOD_LABELS[name]} | {stable[name]['alarm_rate']:.3f} | "
            f"{stable[name]['arl0']:.1f} | {wrong:.0%} | {right:.0%} | {unsure:.0%} | "
            f"{missed:.0%} |"
        )
    body = "\n".join(rows)
    b1 = stable["B1-peeking-t-test"]["alarm_rate"]
    b6 = stable["B6-benchlock"]["alarm_rate"]
    judge_wrong = judge["B1-peeking-t-test"]["verdicts"].get("regression", 0) / headline["seeds"]
    judge_right = judge["B6-benchlock"]["verdicts"].get("judge", 0) / headline["seeds"]

    return f"""## The two headline numbers

### 1. False alarms under peeking

Over {headline["seeds"]} drift-free streams, each inspected after **every one** of
{headline["monitored_runs"]} runs, at `alpha={headline["alpha"]}`:

- **The peeking t-test raises at least one false alarm on {b1:.1%} of healthy pipelines.**
- **Benchlock: {b6:.1%}.**

Peeking is not a misuse of the t-test here. It is the workflow — CI runs on every commit,
and someone looks at the number every time. A test with `alpha = 0.05` re-run at every
accumulating observation has no type-I error control at all.

### 2. Misattribution under a silent judge change

The same pipeline, with the *judge* shifted by 0.10 and the system under test byte-identical:

- **{judge_wrong:.0%} of the time, the peeking t-test reports a regression** — a rollback
  recommendation for a system that never changed.
- **Benchlock reports `judge` {judge_right:.0%} of the time.**

Single-stream methods have no attribution available to them. Monitoring one number, the
only conclusion reachable is "it moved". That is a structural fact about the method, not a
criticism of the people using it — and it is exactly the gap the anchor set fills.

| method | false alarm (drift-free) | ARL₀ | says "regression"/"system" | says `judge` | says `indeterminate` | misses it |
|---|---:|---:|---:|---:|---:|---:|
{body}

    {headline["generated_by"]}
"""


def grid_section(sim: dict[str, Any]) -> str:
    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    delays: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in sim["rows"]:
        method = row["method"]
        if row["truth"] == "stable":
            by_method[method]["false_alarm"].append(row["alarm_rate"])
            if row["arl0"] is not None:
                by_method[method]["arl0"].append(row["arl0"])
        else:
            by_method[method]["judge_as_system"].append(row["judge_as_system"])
            by_method[method]["system_as_judge"].append(row["system_as_judge"])
            by_method[method]["indeterminate"].append(row["indeterminate"])
            if row["detection_rate"] is not None:
                by_method[method]["detection"].append(row["detection_rate"])
            if row["median_delay"] is not None and row["truth"] == "system":
                delays[method][row["system_shift"]].append(row["median_delay"])

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    rows = []
    for name in ORDER:
        stats = by_method.get(name)
        if not stats:
            continue
        rows.append(
            f"| {METHOD_LABELS[name]} | {fmt(mean(stats['false_alarm']))} | "
            f"{fmt(mean(stats['arl0']), '.1f')} | {fmt(mean(stats['detection']), '.2f')} | "
            f"{fmt(mean(stats['judge_as_system']), '.2f')} | "
            f"{fmt(mean(stats['system_as_judge']), '.2f')} | "
            f"{fmt(mean(stats['indeterminate']), '.2f')} |"
        )

    shifts = sorted({s for m in delays.values() for s in m})
    delay_header = " | ".join(f"δs={s:g}" for s in shifts)
    delay_rows = []
    for name in ORDER:
        if name not in delays:
            continue
        cells = " | ".join(fmt(mean(delays[name].get(s, [])), ".1f") for s in shifts)
        delay_rows.append(f"| {METHOD_LABELS[name]} | {cells} |")

    return f"""## Tier 1 — simulation study

{sim["cells"]} cells of a (judge shift × system shift × change point × noise × score type)
grid, {sim["seeds_per_cell"]} streams per cell, horizon {sim["horizon"]} runs.

**Misattribution and detection delay are reported together, always.** Either number alone
is a way to win a benchmark without being useful: report misattribution by itself and a
method that never decides anything wins; report delay by itself and one that fires
constantly wins.

| method | false alarm | ARL₀ | detection rate | judge→system | system→judge | indeterminate |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`judge→system` is how often a method called a judge change a system regression.
`system→judge` is the reverse. For single-stream methods the first column is 1.00 by
construction whenever they fire, because "regression" is the only verdict available to them.

### Median detection delay, in runs

| method | {delay_header} |
|---|{"---:|" * len(shifts)}
{chr(10).join(delay_rows)}

    {sim["generated_by"]}
"""


def antiresult_section(sim: dict[str, Any], headline: dict[str, Any]) -> str:
    b1_delay: list[float] = []
    b6_delay: list[float] = []
    for row in sim["rows"]:
        if row["truth"] != "system" or row["median_delay"] is None:
            continue
        if row["method"] == "B1-peeking-t-test":
            b1_delay.append(row["median_delay"])
        elif row["method"] == "B6-benchlock":
            b6_delay.append(row["median_delay"])

    b1_mean = sum(b1_delay) / len(b1_delay) if b1_delay else float("nan")
    b6_mean = sum(b6_delay) / len(b6_delay) if b6_delay else float("nan")
    b1_fa = headline["stable"]["B1-peeking-t-test"]["alarm_rate"]
    b6_fa = headline["stable"]["B6-benchlock"]["alarm_rate"]

    return f"""## Where Benchlock is worse than a baseline

**Anytime-valid tests are strictly less powerful than fixed-sample tests at the same
nominal n. Benchlock is slower to detect than the invalid peeking t-test, and that is not
a bug to be fixed — it is the price of the guarantee.**

| | B1 peeking t-test | B6 Benchlock |
|---|---:|---:|
| mean median detection delay (runs) | {b1_mean:.1f} | {b6_mean:.1f} |
| false-alarm rate on drift-free streams | {b1_fa:.3f} | {b6_fa:.3f} |

B1 detects a real regression sooner. It also raises a false alarm on {b1_fa:.0%} of
perfectly healthy pipelines, and it cannot tell you whether what moved was your system or
your judge. The extra delay is what buys those two things.

If your situation is one where a false rollback is cheap and a slow detection is
expensive, B1 is the better tool and you should use it. This table is here so that choice
can be made with the numbers in front of you.
"""


def main() -> int:
    headline = load("sim-headline.json")
    sim = load("sim-latest.json")
    if headline is None or sim is None:
        raise SystemExit(
            "missing simulation results. Run:\n"
            "  uv run python bench/sim/run_sim.py --headline --seeds 500 --horizon 150\n"
            "  uv run python bench/sim/run_sim.py --all --seeds 100 --horizon 60"
        )

    real = load("real-latest.json")
    pool = load("real-pool.json")
    attacks = load("adversarial-latest.json")
    nondeterminism = load("judge-nondeterminism.json")
    cost = load("cost.json")

    parts = [
        "<!-- GENERATED by scripts/gen_results.py. Do not edit by hand. -->",
        "# RESULTS",
        "",
        "Every number here comes from a committed run in `bench/results/` and names the "
        "command that produced it. This file is generated; edits to it will be overwritten.",
        "",
        headline_section(headline),
        antiresult_section(sim, headline),
        grid_section(sim),
    ]

    if nondeterminism is not None and not (pool or {}).get("simulated"):
        parts.append(_nondeterminism_section(nondeterminism))
    if real is not None and pool is not None and pool.get("simulated"):
        # Hard Rule 5. A simulated pool cannot answer a question about real judges, and
        # publishing it under a "real judges" heading would be exactly the kind of
        # fabricated number this project exists to make impossible.
        parts.append(_tier2_not_run(real))
    elif real is not None:
        parts.append(_real_section(real))
    if attacks is not None:
        parts.append(_attacks_section(attacks))
    if cost is not None:
        parts.append(
            f"## Cost\n\nTotal spend across every real-judge run: "
            f"**${cost['total_dollars']:.2f}** over {cost['total_calls']:,} judge calls.\n"
        )

    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


def _nondeterminism_section(data: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {r['provider']}/{r['model']} | {r['rubric']} | {r['score_type']} | "
        f"{r['exact_agreement_rate']:.1%} | {r['mean_abs_pairwise_diff']:.4f} | "
        f"{r['run_mean_sd']:.4f} |"
        for r in data["rows"]
    )
    return f"""## How much does a temperature-0 judge disagree with itself?

Measured from K={data["replicates"]} identical calls over {data["n_items"]} items.

| judge | rubric | score type | identical scores | mean |Δ| | run-mean SD |
|---|---|---|---:|---:|---:|
{rows}

    {data["command"]}
"""


def _tier2_not_run(data: dict[str, Any]) -> str:
    """Tier 2 machinery is complete but has not been run against real judges."""
    rows = "\n".join(
        f"| {r['scenario']} | {r['truth']} | {METHOD_LABELS.get(r['method'], r['method'])} | "
        f"{r['verdict']} | {'✅' if r['correct'] else '❌'} |"
        for r in data["rows"]
        if r["method"] in {"B1-peeking-t-test", "B5-benchlock-no-anchor", "B6-benchlock"}
    )
    return f"""## Tier 2 — real judges: NOT RUN

**This section reports no real-judge numbers, because none were measured.** The build
environment had no `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, so no call was ever made to a
hosted judge. Publishing simulated scores under this heading would be precisely the kind of
invented number this project exists to make impossible.

What *is* complete: the provider adapters, the score-pool builder, the stream composer, the
six scenarios with their ground truth, and the runner. The pipeline below was executed
end-to-end against the deterministic built-in judge to prove the code path works. **These
are simulated scores and say nothing about how real judges behave.**

| scenario | ground truth | method | verdict | correct |
|---|---|---|---|---|
{rows}

To run it for real, on a budget of roughly $3-8:

```sh
export ANTHROPIC_API_KEY=...   # and OPENAI_API_KEY for the cross-provider configuration
uv run python bench/real/build_pool.py --items 400 --replicates 5 --dry-run  # cost first
uv run python bench/real/build_pool.py --items 400 --replicates 5
uv run python bench/real/run_real.py --all
uv run python scripts/gen_results.py
```

Two things are unmeasured until then, and both are listed in the README's limitations:

1. **Judge self-disagreement at temperature 0** (Phase 6.5). The most quotable number in
   the project, and it requires a real judge to exist.
2. **Whether a cache-busting nonce perturbs a real judge's scores** (Phase 7.9). The
   mitigation is implemented and its effect is zero against the simulated judge — which is
   an artefact of the simulation, not a finding.
"""


def _real_section(data: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {r['scenario']} | {r['truth']} | {METHOD_LABELS.get(r['method'], r['method'])} | "
        f"{r['verdict']} | {'✅' if r['correct'] else '❌'} |"
        for r in data["rows"]
    )
    return f"""## Tier 2 — real judges

{data["methodology"]}

| scenario | ground truth | method | verdict | correct |
|---|---|---|---|---|
{rows}

    {data["command"]}
"""


def _attacks_section(data: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {r['strategy']} | {r['description']} | {r['failure_rate']:.0%} | {r['status']} |"
        for r in data["rows"]
    )
    return f"""## Attacks that work against Benchlock

These are measured failures, not hypotheticals. Where something was fixed, the pre-fix
number is kept in the table with the commit that changed it.

| attack | what it does | Benchlock's failure rate | status |
|---|---|---:|---|
{rows}

    {data["command"]}
"""


if __name__ == "__main__":
    raise SystemExit(main())
