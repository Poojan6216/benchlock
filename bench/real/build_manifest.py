"""Select the Tier 2 item pool and commit its ids and content hashes.

The dataset itself is never committed — only the ids that identify the items and the
hashes that prove they have not changed. `loader.py` rebuilds the pool from a local copy
and fails loudly if the upstream data has moved underneath the benchmark.

    uv run python bench/real/build_manifest.py --source data/pool.jsonl

Dataset choice, and the reason, are recorded in the manifest so the selection is auditable
rather than a decision that happened once and was forgotten.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bench.real.loader import MANIFEST, ROOT, content_hash

#: Why this dataset. Recorded in the manifest and quoted in RESULTS.md.
DATASETS = {
    "helpsteer2": {
        "name": "HelpSteer2",
        "why": (
            "public, redistributable by reference, >10k (prompt, response) pairs with "
            "human helpfulness ratings, so it has a genuine quality gradient rather than a "
            "synthetic one"
        ),
        "instructions": (
            "download HelpSteer2 and write it to data/pool.jsonl as one JSON object per "
            'line with keys {"item_id", "prompt_input", "output", "tags"}, then re-run '
            "`uv run python bench/real/build_manifest.py`"
        ),
    },
    "custom": {
        "name": "custom",
        "why": "a local pool supplied by the operator",
        "instructions": (
            "write your pool to data/pool.jsonl as one JSON object per line with keys "
            '{"item_id", "prompt_input", "output", "tags"}'
        ),
    },
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "pool.jsonl")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="helpsteer2")
    parser.add_argument("--items", type=int, default=400)
    args = parser.parse_args(argv)

    meta = DATASETS[args.dataset]
    if not args.source.exists():
        raise SystemExit(f"no dataset at {args.source}.\n  fix: {meta['instructions']}")

    records = [
        json.loads(line)
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Deterministic selection: sorted by id, so the manifest does not depend on file order.
    chosen = sorted(records, key=lambda r: str(r["item_id"]))[: args.items]

    MANIFEST.write_text(
        json.dumps(
            {
                "dataset": meta["name"],
                "why_this_dataset": meta["why"],
                "local_path": str(args.source.relative_to(ROOT)),
                "instructions": meta["instructions"],
                "n_items": len(chosen),
                "items": [
                    {
                        "item_id": str(r["item_id"]),
                        "content_hash": content_hash(r["prompt_input"], r["output"]),
                        "tags": list(r.get("tags", [])),
                    }
                    for r in chosen
                ],
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(chosen)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
