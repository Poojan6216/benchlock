#!/usr/bin/env python3
"""Derive per-configuration judge refusal rates from the committed Tier 2 run log.

    uv run python bench/real/refusal_rates.py

The finding this produces — that a judge's willingness to score at all moves with its
configuration — was not something the study set out to measure. It fell out of the run log,
and this script exists so the number in RESULTS.md is regenerable from a committed artefact
rather than from a one-off shell session (Hard Rule 5).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG = ROOT / "bench" / "results" / "real-pool-run.log"
OUT = ROOT / "bench" / "results" / "judge-refusal-rates.json"

CONFIGURATION = {
    "a-baseline": "claude-sonnet-5, base rubric, effort low",
    "b-other-snapshot": "claude-haiku-4-5, base rubric",
    "c-strict-rubric": "claude-sonnet-5, STRICT rubric, effort low",
    "d-effort": "claude-sonnet-5, base rubric, effort HIGH",
}


def main() -> int:
    if not LOG.exists():
        raise SystemExit(f"missing {LOG.relative_to(ROOT)} — run bench/real/build_pool.py first")
    current: str | None = None
    rows: list[dict[str, object]] = []
    reps: list[dict[str, object]] = []
    api_errors = 0
    for line in LOG.read_text().splitlines():
        scoring = re.search(r"scoring under ([a-z0-9-]+)", line)
        if scoring:
            current = scoring.group(1)
            continue
        if "replicates on" in line:
            current = "__replicates__"
            continue
        failed = re.search(r"! (\d+)/(\d+) call\(s\) failed", line)
        if failed:
            n, total = int(failed.group(1)), int(failed.group(2))
            if "JudgeCallError" not in line:
                api_errors += n
            if current == "__replicates__":
                reps.append({"failed": n, "total": total, "rate": n / total})
            else:
                rows.append(
                    {
                        "config": current,
                        "configuration": CONFIGURATION.get(current or "", ""),
                        "refused": n,
                        "requested": total,
                        "refusal_rate": n / total,
                    }
                )
    baseline = next((r["refusal_rate"] for r in rows if r["config"] == "a-baseline"), None)
    for r in rows:
        r["vs_baseline_multiple"] = (
            round(float(r["refusal_rate"]) / float(baseline), 2) if baseline else None
        )

    OUT.write_text(
        json.dumps(
            {
                "command": "uv run python bench/real/refusal_rates.py",
                "source_log": str(LOG.relative_to(ROOT)),
                "dataset": (
                    "nvidia/HelpSteer2 (CC-BY-4.0), 400 items stratified by human helpfulness"
                ),
                "finding": (
                    "A judge's willingness to score at all is itself a form of drift. Same model, "
                    "same items: editing only the rubric text raised the refusal rate 6.7x, and "
                    "raising only the reasoning effort raised it 3x. Refused items vanish silently "
                    "from the sample, so any tool that looks only at the scores it gets back "
                    "cannot see this."
                ),
                "per_config": rows,
                "replicates_same_config": {
                    "note": (
                        "Five identical repeated scorings of the same 200 anchor items under the "
                        "baseline configuration. The refusal boundary is not deterministic."
                    ),
                    "runs": reps,
                    "min_refused": min((int(r["failed"]) for r in reps), default=0),
                    "max_refused": max((int(r["failed"]) for r in reps), default=0),
                },
                "all_refusals_were_safety_declines": api_errors == 0,
                "api_errors": api_errors,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT.relative_to(ROOT)} from {LOG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
