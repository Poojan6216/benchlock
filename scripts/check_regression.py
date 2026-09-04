#!/usr/bin/env python3
"""Nightly guard: fail the build if a headline metric has regressed.

The benchmark is a test, not a marketing artefact. Thresholds are deliberately loose
enough to absorb Monte-Carlo noise and tight enough that a real degradation is caught.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"

#: (description, check) — each returns an error string, or None when healthy.
THRESHOLDS = {
    "benchlock false-alarm rate on drift-free streams": 0.05,
    "benchlock correct-attribution rate under a silent judge change": 0.90,
}


def main() -> int:
    path = RESULTS / "sim-headline.json"
    if not path.exists():
        print(f"missing {path.relative_to(ROOT)} — run the headline study first")
        return 1
    data = json.loads(path.read_text())
    failures: list[str] = []

    false_alarm = data["stable"]["B6-benchlock"]["alarm_rate"]
    limit = THRESHOLDS["benchlock false-alarm rate on drift-free streams"]
    if false_alarm > limit:
        failures.append(
            f"false-alarm rate {false_alarm:.3f} exceeds alpha={limit}. The guarantee is "
            "the product; this is not a threshold to relax"
        )

    verdicts = data["judge"]["B6-benchlock"]["verdicts"]
    total = sum(verdicts.values()) or 1
    correct = verdicts.get("judge", 0) / total
    floor = THRESHOLDS["benchlock correct-attribution rate under a silent judge change"]
    if correct < floor:
        failures.append(
            f"correct attribution under a silent judge change fell to {correct:.0%}, "
            f"below the committed floor of {floor:.0%}"
        )

    # An `indeterminate` is never a failure — refusing to guess is the design — but a
    # `system` verdict on a judge-only stream is the exact error the tool exists to prevent.
    wrong = verdicts.get("system", 0) / total
    if wrong > 0.01:
        failures.append(
            f"{wrong:.0%} of judge-only streams produced a `system` verdict — a confident, "
            "wrong rollback recommendation"
        )

    if failures:
        print("headline metrics regressed:\n")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"headline metrics healthy: false alarm {false_alarm:.3f}, "
        f"correct attribution {correct:.0%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
