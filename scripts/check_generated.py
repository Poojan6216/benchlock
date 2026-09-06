#!/usr/bin/env python3
"""Every generated document must match its generator, byte for byte.

README.md, RESULTS.md and the docs under `docs/` are produced from `bench/results/` by the
`gen_*.py` scripts and `bench/demo.py`. A hand edit to any of them — or a generator change
whose output was never re-run — is how prose and measurement drift apart while both stay
plausible. This script regenerates all of them and fails if the working tree changed.

Plots are deliberately not covered: PNG bytes vary with the matplotlib build, so a
byte-level diff would fail on every machine but the one that made them.

    uv run python scripts/check_generated.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: In dependency order: demo.md is lifted into README.md, and summary.json (written by
#: gen_results.py) is read by gen_writeup.py.
GENERATORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bench/demo.py", ("docs/demo.md",)),
    ("scripts/gen_results.py", ("RESULTS.md", "bench/results/summary.json")),
    ("scripts/gen_guarantees.py", ("docs/statistical-guarantees.md",)),
    ("scripts/gen_threat_model.py", ("docs/threat-model.md",)),
    ("scripts/gen_writeup.py", ("docs/writeup.md",)),
    ("scripts/gen_readme.py", ("README.md",)),
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def main() -> int:
    outputs = [path for _, paths in GENERATORS for path in paths]
    if git("status", "--porcelain", "--", *outputs).strip():
        print("refusing to run: generated files already have uncommitted changes:")
        print(git("status", "--porcelain", "--", *outputs))
        print("commit or revert them first, so a diff here means the generator disagrees")
        return 2

    for script, _ in GENERATORS:
        run = subprocess.run(
            [sys.executable, str(ROOT / script)], cwd=ROOT, capture_output=True, text=True
        )
        if run.returncode != 0:
            print(f"{script} failed:\n{run.stdout}{run.stderr}")
            return 1

    diff = git("diff", "--stat", "--", *outputs).strip()
    if diff:
        print("generated documents disagree with their generators:\n")
        print(diff)
        print("\nregenerate and commit, or revert the hand edit:")
        for script, paths in GENERATORS:
            print(f"  uv run python {script:<32s} -> {', '.join(paths)}")
        return 1

    print(f"ok: {len(outputs)} generated files match their generators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
