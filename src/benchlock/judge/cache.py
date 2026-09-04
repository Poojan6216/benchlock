"""A local score cache, and the nonce that defeats a *provider's* cache.

Two different caches, pulling in opposite directions, and the distinction matters:

* **Ours** exists to stop a benchmark re-paying for scores it already has. It is keyed on
  everything that could change the answer, so a changed rubric or model is a miss.
* **The provider's** is a hazard. If a hosted judge serves a cached response for an
  identical prompt, the anchor set returns *yesterday's answers*, the anchor stream looks
  perfectly stable, and real judge drift becomes invisible — the tool fails silently in
  the one direction it must not. A per-run nonce in the prompt defeats it.

The nonce is itself a change to the prompt, so it could perturb the score it was meant to
preserve. That is measured rather than assumed: Phase 7.9 quantifies it, and if a nonce
moves scores the trade-off is documented rather than shipped quietly.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from benchlock.judge.base import (
    CostEstimate,
    JudgeAdapter,
    JudgeDescription,
    JudgeRequest,
    JudgeResult,
)
from benchlock.model.pins import JudgePin, sha256_text

DEFAULT_CACHE_PATH = Path(".benchlock/judge-cache.sqlite")


def cache_key(pin: JudgePin, request: JudgeRequest) -> str:
    """Everything that could change the score. A miss is cheaper than a wrong hit."""
    return sha256_text(
        json.dumps(
            {
                "provider": pin.provider,
                "model": pin.model,
                "rubric": pin.rubric_hash,
                "params": pin.params_hash,
                "item": request.item_id,
                "input": request.prompt_input,
                "output": request.output,
                "nonce": request.nonce,
            },
            sort_keys=True,
        )
    )


class ScoreCache:
    """SQLite-backed score cache. Holds eval content, so it lives under `.benchlock/`."""

    def __init__(self, path: Path = DEFAULT_CACHE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS scores ("
                "  key TEXT PRIMARY KEY,"
                "  raw_score REAL NOT NULL,"
                "  input_tokens INTEGER NOT NULL DEFAULT 0,"
                "  output_tokens INTEGER NOT NULL DEFAULT 0"
                ")"
            )
            db.commit()

    def get(self, key: str) -> JudgeResult | None:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT raw_score, input_tokens, output_tokens FROM scores WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return JudgeResult("", float(row[0]), int(row[1]), int(row[2]), cached=True)

    def put(self, key: str, result: JudgeResult) -> None:
        with closing(sqlite3.connect(self.path)) as db:
            db.execute(
                "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
                (key, result.raw_score, result.input_tokens, result.output_tokens),
            )
            db.commit()

    def __len__(self) -> int:
        with closing(sqlite3.connect(self.path)) as db:
            return int(db.execute("SELECT COUNT(*) FROM scores").fetchone()[0])


@dataclass
class CachedJudge:
    """Wraps any adapter with our local cache. The provider's cache is a separate problem."""

    inner: JudgeAdapter
    cache: ScoreCache

    def describe(self) -> JudgeDescription:
        return self.inner.describe()

    def pin(self) -> JudgePin:
        return self.inner.pin()

    def estimate_cost(self, n_items: int, avg_input_tokens: int = 600) -> CostEstimate:
        return self.inner.estimate_cost(n_items, avg_input_tokens)

    def score(self, requests: Sequence[JudgeRequest]) -> Sequence[JudgeResult]:
        pin = self.inner.pin()
        results: dict[str, JudgeResult] = {}
        missing: list[JudgeRequest] = []
        for request in requests:
            hit = self.cache.get(cache_key(pin, request))
            if hit is None:
                missing.append(request)
            else:
                results[request.item_id] = JudgeResult(
                    request.item_id, hit.raw_score, hit.input_tokens, hit.output_tokens, cached=True
                )
        for request, fresh in zip(missing, self.inner.score(missing), strict=True):
            self.cache.put(cache_key(pin, request), fresh)
            results[request.item_id] = fresh
        return [results[r.item_id] for r in requests]
