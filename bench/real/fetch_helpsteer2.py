#!/usr/bin/env python3
"""Download a stratified item pool from HelpSteer2.

HelpSteer2 (nvidia/HelpSteer2, CC-BY-4.0) is a public set of (prompt, response) pairs with
**human** helpfulness ratings on a 0-4 scale. That human rating is what makes it the right
dataset here: it supplies a genuine quality gradient rather than a synthetic one, so a
judge scoring these items has something real to disagree about.

The data is written to `data/pool.jsonl`, which is gitignored. Only the item ids and
content hashes are committed, in `bench/real/pool-manifest.json` — redistribution by
reference, per spec 6.1.

    uv run python bench/real/fetch_helpsteer2.py --items 400
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "pool.jsonl"
API = "https://datasets-server.huggingface.co/rows"
DATASET = "nvidia/HelpSteer2"

#: Cap on characters kept per field. HelpSteer2 responses run long, and a judge call's cost
#: is dominated by input tokens; truncating at a generous bound keeps the study affordable
#: without flattening the quality gradient. Recorded in the manifest so it is auditable.
MAX_PROMPT_CHARS = 1200
MAX_RESPONSE_CHARS = 2400


def fetch(offset: int, length: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{API}?{query}", timeout=60) as response:
        payload = json.load(response)
    if "error" in payload:
        raise SystemExit(f"HuggingFace returned an error: {payload['error']}")
    return [r["row"] for r in payload["rows"]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=400)
    parser.add_argument("--scan", type=int, default=1000, help="rows to scan before sampling")
    args = parser.parse_args(argv)

    print(f"scanning {args.scan} rows of {DATASET}...")
    rows: list[dict] = []
    for offset in range(0, args.scan, 100):
        rows.extend(fetch(offset, min(100, args.scan - offset)))
        print(f"  {len(rows)} rows", flush=True)

    # Stratify by the human helpfulness rating so the pool spans the quality range rather
    # than piling up wherever the dataset happens to be dense.
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("prompt") and row.get("response"):
            buckets[int(row["helpfulness"])].append(row)

    per_bucket = args.items // max(1, len(buckets))
    chosen: list[dict] = []
    for rating in sorted(buckets):
        chosen.extend(buckets[rating][:per_bucket])
    # Top up deterministically from the largest buckets if rounding left a shortfall.
    for rating in sorted(buckets, key=lambda r: -len(buckets[r])):
        if len(chosen) >= args.items:
            break
        for row in buckets[rating][per_bucket:]:
            if len(chosen) >= args.items:
                break
            chosen.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for index, row in enumerate(chosen):
            fh.write(
                json.dumps(
                    {
                        "item_id": f"helpsteer2-{index:04d}",
                        "prompt_input": row["prompt"][:MAX_PROMPT_CHARS],
                        "output": row["response"][:MAX_RESPONSE_CHARS],
                        # The human rating, kept for `human` anchor mode and for checking
                        # whether the judge tracks human judgement at all.
                        "human_helpfulness": int(row["helpfulness"]),
                        "tags": [f"helpfulness-{int(row['helpfulness'])}"],
                    }
                )
                + "\n"
            )

    spread = {r: sum(1 for c in chosen if int(c["helpfulness"]) == r) for r in sorted(buckets)}
    chars = [
        len(c["prompt"][:MAX_PROMPT_CHARS]) + len(c["response"][:MAX_RESPONSE_CHARS])
        for c in chosen
    ]
    print(f"\nwrote {len(chosen)} items to {OUT.relative_to(ROOT)}")
    print(f"  helpfulness spread: {spread}")
    mean_chars = sum(chars) / len(chars)
    print(f"  mean chars/item: {mean_chars:.0f}  (~{mean_chars / 3.8:.0f} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
