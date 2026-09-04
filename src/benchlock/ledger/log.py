"""The append-only, hash-chained ledger.

Every record carries the SHA-256 of the record before it, so the file is tamper-evident:
editing any byte of any record breaks the chain from that point on, and `verify` reports
the index of the first record that fails.

**Hard Rule 9 is structural here, not aspirational.** Records are built by an explicit
allowlist (`_run_payload`), so there is no code path by which a prompt, a model output or
a judge rationale can reach the file — not because we remember to strip them, but because
nothing ever copies them in. Item ids are stored *hashed* for the same reason: in real
eval suites an item id is frequently the prompt text, or contains a customer identifier.
The statistics need stable identity, not the original string.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from ulid import ULID

from benchlock import DECISION_SEMANTICS_VERSION
from benchlock.model.pins import AnchorPin, JudgePin, sha256_json, sha256_text
from benchlock.model.streams import Observation, RunRecord, StreamKind

LEDGER_SCHEMA = 1

#: Domain-separated genesis link, so a ledger cannot be spliced onto another file.
GENESIS_HASH = sha256_text("benchlock-ledger-genesis-v1")

DEFAULT_LEDGER_PATH = Path(".benchlock/ledger.jsonl")


class RecordType(StrEnum):
    RUN = "run"  # one CI run of one stream
    BASELINE = "baseline"  # the anchor set was frozen and the noise floor measured
    REBASELINE = "rebaseline"  # an explicit, reasoned epoch boundary (Hard Rule 8)


class LedgerError(Exception):
    """Raised for a broken chain, a malformed record, or an out-of-order append."""

    def __init__(self, message: str, *, index: int | None = None, hint: str = "") -> None:
        self.index = index
        self.hint = hint
        where = f" at record {index}" if index is not None else ""
        #: The problem without the fix, so a renderer can lay the two out itself.
        self.message = f"{message}{where}"
        super().__init__(self.message + (f"\n  fix: {hint}" if hint else ""))


def hash_item_id(item_id: str) -> str:
    """Stable, content-free identity for an eval item (Hard Rule 9).

    Truncated to 32 hex chars: 128 bits, far beyond collision risk for eval-suite sizes,
    and short enough to keep the ledger readable.
    """
    return sha256_text(f"benchlock-item-v1:{item_id}")[:32]


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    """One line of the ledger, with its place in the chain."""

    seq: int
    type: RecordType
    prev: str
    payload: Mapping[str, Any]
    semantics_version: int
    schema: int
    hash: str

    def body(self) -> dict[str, Any]:
        """The hashed portion: everything except the hash itself."""
        return {
            "seq": self.seq,
            "type": self.type.value,
            "prev": self.prev,
            "semantics_version": self.semantics_version,
            "schema": self.schema,
            "payload": dict(self.payload),
        }

    def compute_hash(self) -> str:
        return sha256_json(self.body())

    def to_line(self) -> str:
        return json.dumps({**self.body(), "hash": self.hash}, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> LedgerRecord:
        return cls(
            seq=int(data["seq"]),
            type=RecordType(data["type"]),
            prev=str(data["prev"]),
            payload=dict(data["payload"]),
            semantics_version=int(data["semantics_version"]),
            schema=int(data["schema"]),
            hash=str(data["hash"]),
        )


# ---------------------------------------------------------------------------------------
# Payload builders — the Hard Rule 9 allowlist
# ---------------------------------------------------------------------------------------

#: The only keys a run payload may contain. Enforced on write AND asserted in tests.
RUN_PAYLOAD_KEYS = frozenset(
    {
        "run_id",
        "run_index",
        "kind",
        "suite_hash",
        "judge_pin",
        "anchor_pin",
        "epoch",
        "observations",
    }
)
OBSERVATION_KEYS = frozenset({"item", "score", "raw"})


def _run_payload(run: RunRecord) -> dict[str, Any]:
    """Project a RunRecord onto exactly the fields the ledger is allowed to hold.

    This function is the boundary. Nothing here reads free text from the eval, so no
    prompt, output or rationale can reach the file even if one is attached upstream.
    """
    return {
        "run_id": run.run_id,
        "run_index": run.run_index,
        "kind": run.kind.value,
        "suite_hash": run.suite_hash,
        "judge_pin": run.judge_pin.to_json(),
        "anchor_pin": run.anchor_pin.to_json() if run.anchor_pin is not None else None,
        "epoch": run.epoch,
        "observations": [
            {"item": hash_item_id(o.item_id), "score": o.score, "raw": o.raw_score}
            for o in run.observations
        ],
    }


def _check_allowlist(payload: Mapping[str, Any]) -> None:
    extra = set(payload) - RUN_PAYLOAD_KEYS
    if extra:
        raise LedgerError(
            f"refusing to write unknown ledger fields: {', '.join(sorted(extra))}",
            hint="Hard Rule 9: the ledger holds scores, hashes and metadata only",
        )
    for obs in payload.get("observations", []):
        obs_extra = set(obs) - OBSERVATION_KEYS
        if obs_extra:
            raise LedgerError(
                f"refusing to write unknown observation fields: {', '.join(sorted(obs_extra))}",
                hint="Hard Rule 9: an observation is an id hash and two numbers",
            )


# ---------------------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------------------


@contextmanager
def _file_lock(path: Path, timeout: float = 10.0) -> Iterator[None]:
    """Exclusive advisory lock, so parallel CI jobs cannot interleave records.

    Hardened and stress-tested in Phase 8.4. On a platform without fcntl we proceed
    without a lock rather than pretending to hold one.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return

    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + timeout
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LedgerError(
                        f"could not lock {lock_path} within {timeout:g}s",
                        hint=(
                            "another benchlock process is writing the ledger; "
                            "serialise your CI jobs or give each one its own ledger"
                        ),
                    ) from None
                time.sleep(0.01)
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------------------


class Ledger:
    """Append-only JSONL with a SHA-256 chain. Reads verify; writes extend."""

    def __init__(self, path: Path = DEFAULT_LEDGER_PATH) -> None:
        self.path = path

    # ---- reading ----------------------------------------------------------------------

    def exists(self) -> bool:
        return self.path.exists()

    def read_raw(self) -> list[LedgerRecord]:
        """Parse every record without checking the chain."""
        if not self.path.exists():
            return []
        records: list[LedgerRecord] = []
        for line_no, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LedgerError(
                    f"ledger line {line_no} is not valid JSON: {exc.msg}",
                    index=len(records),
                    hint="the ledger is append-only and machine-written; restore it from git",
                ) from exc
            try:
                records.append(LedgerRecord.from_json(data))
            except (KeyError, ValueError) as exc:
                raise LedgerError(
                    f"ledger line {line_no} is missing required fields: {exc}",
                    index=len(records),
                    hint="the ledger is append-only and machine-written; restore it from git",
                ) from exc
        return records

    def verify(self) -> list[LedgerRecord]:
        """Read and check the whole chain. Raises naming the first broken record."""
        records = self.read_raw()
        prev = GENESIS_HASH
        for expected_seq, record in enumerate(records):
            if record.seq != expected_seq:
                raise LedgerError(
                    f"ledger sequence jumped: expected seq {expected_seq}, found {record.seq}",
                    index=expected_seq,
                    hint="a record was inserted, removed or reordered; the ledger is append-only",
                )
            if record.prev != prev:
                raise LedgerError(
                    "ledger chain broken: this record's `prev` does not match the previous "
                    f"record's hash (expected {prev[:12]}…, found {record.prev[:12]}…)",
                    index=expected_seq,
                    hint="a record was edited, inserted or removed; restore the ledger from git",
                )
            recomputed = record.compute_hash()
            if recomputed != record.hash:
                raise LedgerError(
                    f"ledger record was modified: content hashes to {recomputed[:12]}… but "
                    f"the record claims {record.hash[:12]}…",
                    index=expected_seq,
                    hint="a record's contents were edited in place; restore the ledger from git",
                )
            prev = record.hash
        return records

    def head(self) -> tuple[int, str]:
        """(next seq, hash to chain from). Verifies the chain first."""
        records = self.verify()
        if not records:
            return 0, GENESIS_HASH
        return records[-1].seq + 1, records[-1].hash

    # ---- writing ----------------------------------------------------------------------

    def _append(self, record_type: RecordType, payload: Mapping[str, Any]) -> LedgerRecord:
        with _file_lock(self.path):
            seq, prev = self.head()
            record = LedgerRecord(
                seq=seq,
                type=record_type,
                prev=prev,
                payload=dict(payload),
                semantics_version=DECISION_SEMANTICS_VERSION,
                schema=LEDGER_SCHEMA,
                hash="",
            )
            record = replace(record, hash=record.compute_hash())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(record.to_line() + "\n")
            return record

    def append_run(self, run: RunRecord) -> LedgerRecord:
        payload = _run_payload(run)
        _check_allowlist(payload)
        return self._append(RecordType.RUN, payload)

    def append_baseline(
        self,
        *,
        judge_pin: JudgePin,
        anchor_pin: AnchorPin,
        epoch: int = 0,
        note: str = "",
    ) -> LedgerRecord:
        return self._append(
            RecordType.BASELINE,
            {
                "epoch": epoch,
                "judge_pin": judge_pin.to_json(),
                "anchor_pin": anchor_pin.to_json(),
                "note": note[:200],
            },
        )

    def append_rebaseline(
        self,
        *,
        reason: str,
        epoch: int,
        judge_pin: JudgePin,
        anchor_pin: AnchorPin | None,
        pin_delta: Sequence[str] = (),
    ) -> LedgerRecord:
        if not reason.strip():
            raise LedgerError(
                "rebaseline requires a reason",
                hint="run `benchlock rebaseline --reason judge-version-change`",
            )
        return self._append(
            RecordType.REBASELINE,
            {
                "epoch": epoch,
                "reason": reason.strip()[:500],
                "judge_pin": judge_pin.to_json(),
                "anchor_pin": anchor_pin.to_json() if anchor_pin is not None else None,
                "pin_delta": list(pin_delta),
            },
        )

    # ---- reconstruction ---------------------------------------------------------------

    def runs(self, kind: StreamKind | None = None) -> list[RunRecord]:
        """Rebuild RunRecords from the ledger. This is what `replay` decides over."""
        out: list[RunRecord] = []
        for record in self.verify():
            if record.type is not RecordType.RUN:
                continue
            payload = record.payload
            run_kind = StreamKind(payload["kind"])
            if kind is not None and run_kind is not kind:
                continue
            anchor_pin = payload.get("anchor_pin")
            judge_pin = JudgePin.from_json(payload["judge_pin"])
            out.append(
                RunRecord(
                    run_id=str(payload["run_id"]),
                    run_index=int(payload["run_index"]),
                    kind=run_kind,
                    observations=tuple(
                        Observation(
                            item_id=str(o["item"]),
                            score=float(o["score"]),
                            raw_score=float(o["raw"]),
                            scale=judge_pin.scale,
                        )
                        for o in payload["observations"]
                    ),
                    suite_hash=str(payload["suite_hash"]),
                    judge_pin=judge_pin,
                    anchor_pin=AnchorPin.from_json(anchor_pin) if anchor_pin else None,
                    epoch=int(payload.get("epoch", 0)),
                )
            )
        return out

    def epoch(self) -> int:
        """The current baseline epoch: number of rebaselines so far."""
        return sum(1 for r in self.read_raw() if r.type is RecordType.REBASELINE)


def new_run_id() -> str:
    """A fresh ULID. The only nondeterminism in the write path; never read by `decide`."""
    return str(ULID())
