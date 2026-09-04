"""Reconstruct the Tier 2 item pool from a committed manifest of ids.

The dataset itself is never committed: the manifest holds ids and content hashes, and this
module rebuilds the exact item set from them. That keeps the repository free of
redistributed data while leaving the benchmark reproducible — the hashes fail loudly if the
upstream data ever changes underneath us.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "bench" / "real" / "pool-manifest.json"


@dataclass(frozen=True, slots=True)
class PoolItem:
    item_id: str
    prompt_input: str
    output: str
    tags: tuple[str, ...]
    dataset: str


class DatasetError(Exception):
    def __init__(self, message: str, hint: str = "") -> None:
        self.message = message
        self.hint = hint
        super().__init__(f"{message}" + (f"\n  fix: {hint}" if hint else ""))


def content_hash(prompt_input: str, output: str) -> str:
    return hashlib.sha256(f"{prompt_input}\x00{output}".encode()).hexdigest()[:16]


def load_items(limit: int | None = None) -> list[PoolItem]:
    """Rebuild the pool. Raises with instructions if the dataset is not available locally."""
    if not MANIFEST.exists():
        raise DatasetError(
            f"no pool manifest at {MANIFEST.relative_to(ROOT)}",
            "run `uv run python bench/real/build_manifest.py` to select an item set and "
            "record its ids and hashes",
        )
    manifest = json.loads(MANIFEST.read_text())
    source = ROOT / manifest["local_path"]
    if not source.exists():
        raise DatasetError(
            f"the dataset is not present at {manifest['local_path']}",
            f"{manifest['instructions']}",
        )

    by_id: dict[str, dict[str, str]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            by_id[str(record["item_id"])] = record

    items: list[PoolItem] = []
    for entry in manifest["items"][:limit]:
        record = by_id.get(entry["item_id"])
        if record is None:
            raise DatasetError(
                f"item {entry['item_id']!r} from the manifest is missing from the dataset",
                "the upstream data has changed; regenerate the manifest and re-run the pool",
            )
        actual = content_hash(record["prompt_input"], record["output"])
        if actual != entry["content_hash"]:
            raise DatasetError(
                f"item {entry['item_id']!r} no longer matches its committed hash",
                "the upstream data changed underneath the benchmark; regenerate the "
                "manifest and say so in RESULTS.md rather than comparing across the change",
            )
        items.append(
            PoolItem(
                item_id=entry["item_id"],
                prompt_input=record["prompt_input"],
                output=record["output"],
                tags=tuple(entry.get("tags", ())),
                dataset=manifest["dataset"],
            )
        )
    return items
