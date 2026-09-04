#!/usr/bin/env python3
"""Benchlock gating its own repository, in CI, on its own fixture streams.

Builds a real ledger from a committed golden stream, runs `verdict`, `gate`, `report` and
`replay` against it, and asserts the verdicts are the ones the fixtures say they should be.
If the tool cannot survive its own CI, it has no business in anyone else's.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.sim.generate import generate, read_manifest, spec_from_json  # noqa: E402

from benchlock.ledger.log import Ledger  # noqa: E402

CONFIG = """\
version: 1
alpha: 0.05
score_scale: [1, 5]
min_runs: 8
min_obs: 30
judge:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  rubric: ./rubric.md
gate:
  fail_on: [system, both]
  warn_on: [indeterminate]
"""

#: (fixture, expected verdict, expected exit code under the default gate)
CASES = [
    ("demo1-phantom-judge", "judge", 0),
    ("demo2-real-regression", "system", 1),
    ("demo3-under-provisioned", "indeterminate", 2),
    ("stable-control", "stable", 0),
]


def build(root: Path, fixture: str) -> None:
    entry = next(
        e
        for e in read_manifest(ROOT / "tests" / "fixtures" / "streams" / "manifest.json")
        if e["name"] == fixture
    )
    system, anchor = generate(spec_from_json(entry))
    (root / "rubric.md").write_text("Score the answer 1-5 for helpfulness and factual accuracy.\n")
    (root / "benchlock.yaml").write_text(CONFIG)
    book = Ledger(root / ".benchlock" / "ledger.jsonl")
    book.append_baseline(
        judge_pin=anchor[0].judge_pin, anchor_pin=anchor[0].anchor_pin, note="dogfood"
    )
    for s, a in zip(system, anchor, strict=True):
        book.append_run(s)
        book.append_run(a)


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "benchlock",
            *args,
            "--config",
            str(root / "benchlock.yaml"),
            "--ledger",
            str(root / ".benchlock" / "ledger.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    failures: list[str] = []
    for fixture, expected_verdict, expected_code in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build(root, fixture)

            verdict = run(root, "verdict")
            if f"verdict={expected_verdict}" not in verdict.stdout:
                failures.append(f"{fixture}: expected verdict={expected_verdict}\n{verdict.stdout}")
            gate = run(root, "gate")
            if gate.returncode != expected_code:
                failures.append(
                    f"{fixture}: gate exited {gate.returncode}, expected {expected_code}"
                )
            report = run(root, "report")
            if not report.stdout.startswith("### "):
                failures.append(f"{fixture}: report produced no markdown")
            replay = run(root, "replay")
            if replay.returncode != 0:
                failures.append(f"{fixture}: replay failed\n{replay.stdout}")
            print(
                f"  {fixture:<26s} verdict={expected_verdict:<14s} "
                f"gate={gate.returncode}  replay=ok"
            )

    if failures:
        print("\ndogfood FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nbenchlock gates its own repository correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
