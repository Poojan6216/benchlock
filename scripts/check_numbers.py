#!/usr/bin/env python3
"""Hard Rule 5: every number in README.md and RESULTS.md must trace to a committed run.

Greps both documents for numeric literals and fails on any that cannot be found in
`bench/results/`. This is the check that makes "we never report a number we didn't
measure" enforceable rather than aspirational — a claim in a README is exactly where an
invented number would hide.

    uv run python scripts/check_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
DOCS = ("README.md", "RESULTS.md")

#: Numbers that are structural rather than measured: version strings, alpha, list markers,
#: years, and the small integers that appear in prose ("three demos", "five verdicts").
EXEMPT = {
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "0.05",  # alpha, declared in the config
    "1.0",
    "100",
    "1939",  # Ville
    "2023",
    "2024",
    "2025",
    "2026",
}

NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


def measured_values() -> set[str]:
    """Every number appearing anywhere in the committed results, in several renderings."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            found.add(f"{node}")
            found.add(f"{node:g}")
            for places in range(5):
                found.add(f"{node:.{places}f}")
            if isinstance(node, float) and 0.0 <= node <= 1.0:
                # Percentages, as rendered in prose and tables.
                for places in range(3):
                    found.add(f"{node * 100:.{places}f}")
                    found.add(f"{node * 100:.{places}f}".rstrip("0").rstrip("."))
            found.add(f"{int(node):,}" if abs(node) >= 1000 else "")
        elif isinstance(node, str):
            for match in NUMBER.finditer(node):
                found.add(match.group(1))

    for path in sorted(RESULTS.glob("*.json")):
        walk(json.loads(path.read_text()))
    found.discard("")
    return found


def main() -> int:
    if not RESULTS.exists() or not list(RESULTS.glob("*.json")):
        print("no committed results to check against — run the benchmarks first")
        return 1

    known = measured_values() | EXEMPT
    failures: list[str] = []

    for name in DOCS:
        path = ROOT / name
        if not path.exists():
            continue
        in_fence = False
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            # Code fences hold commands and captured transcripts, which are verified by
            # the demo regeneration check rather than by number tracing.
            if in_fence or line.lstrip().startswith(("|---", "<!--", "    ")):
                continue
            for match in NUMBER.finditer(line):
                value = match.group(1)
                if value in known:
                    continue
                # Trailing-zero renderings: 0.212 measured, "21.2%" written.
                if value.rstrip("0").rstrip(".") in known:
                    continue
                failures.append(f"{name}:{line_no}: {value!r} — {line.strip()[:88]}")

    if failures:
        print(f"{len(failures)} number(s) in the docs do not trace to a committed run:\n")
        for failure in failures:
            print(f"  {failure}")
        print(
            "\nHard Rule 5: every figure in README.md and RESULTS.md must come from a run "
            "you actually executed, with the command committed in the repo."
        )
        return 1

    print(f"all numbers in {', '.join(DOCS)} trace to committed runs in bench/results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
