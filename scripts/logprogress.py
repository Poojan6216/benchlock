#!/usr/bin/env python3
"""Append one line to the Progress Log fence in BUILD_SPEC.md and tick its checkbox.

Usage: python scripts/logprogress.py "[0.1] what I did — decisions or blockers"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parent.parent / "BUILD_SPEC.md"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: logprogress.py '[phase.task] message'", file=sys.stderr)
        return 2
    line = sys.argv[1].strip()
    text = SPEC.read_text()

    # 1. Append inside the Progress Log fenced block.
    marker = "## Progress Log"
    head, sep, tail = text.partition(marker)
    if not sep:
        print("Progress Log section not found", file=sys.stderr)
        return 1
    fence = re.search(r"```\n(.*?)```", tail, flags=re.S)
    if fence is None:
        print("Progress Log fence not found", file=sys.stderr)
        return 1
    body = fence.group(1)
    if line in body:
        print(f"already logged: {line}")
        return 0
    new_body = body + line + "\n"
    tail = tail[: fence.start()] + "```\n" + new_body + "```" + tail[fence.end() :]

    # 2. Tick the matching task checkbox, e.g. "- [ ] **0.1 — ...".
    task_id = re.match(r"\[([0-9]+\.[0-9]+)\]", line)
    if task_id:
        pat = re.compile(r"- \[ \] (\*\*" + re.escape(task_id.group(1)) + r" —)")
        head_new, n_head = pat.subn(r"- [x] \1", head)
        if n_head:
            head = head_new
        else:
            tail = pat.sub(r"- [x] \1", tail)

    SPEC.write_text(head + sep + tail)
    print(f"logged: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
