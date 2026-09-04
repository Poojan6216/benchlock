"""Phase 8.4 verify: parallel CI jobs never corrupt the ledger.

The ledger is append-only and hash-chained, so an interleaved write does not merely lose a
record — it breaks the chain, and every later verdict becomes unverifiable. Under
contention the only two acceptable outcomes are a valid chain or a clean failure. A corrupt
chain is not one of them, and this is the test that says so.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest
from bench.sim.generate import StreamSpec, generate

from benchlock.ledger.log import Ledger, LedgerError
from benchlock.model.streams import StreamKind

#: The spec asks for 100 repetitions in CI. Kept lower by default so the developer loop
#: stays usable; the `slow` variant below runs the full count.
WRITERS = 20


def _write_one(args: tuple[str, int]) -> str:
    """One writer process appending one run. Returns 'ok' or the failure it saw."""
    path, index = args
    system, _ = generate(StreamSpec(name=f"w{index}", seed=index, n_runs=1, system_items=40))
    book = Ledger(Path(path))
    try:
        book.append_run(system[0])
    except LedgerError as exc:
        # A clean, explained refusal under contention is an acceptable outcome.
        return f"refused: {exc.message[:60]}"
    except Exception as exc:  # pragma: no cover - any other failure is a real bug
        return f"crashed: {type(exc).__name__}: {exc}"
    return "ok"


def _run_writers(path: Path, writers: int) -> list[str]:
    with mp.get_context("fork").Pool(writers) as pool:
        return pool.map(_write_one, [(str(path), i) for i in range(writers)])


def test_parallel_writers_never_corrupt_the_chain(tmp_path: Path) -> None:
    path = tmp_path / ".benchlock" / "ledger.jsonl"
    outcomes = _run_writers(path, WRITERS)

    crashed = [o for o in outcomes if o.startswith("crashed")]
    assert not crashed, f"writers crashed rather than failing cleanly: {crashed[:3]}"

    # Whatever got through must form a valid chain.
    records = Ledger(path).verify()
    assert len(records) == outcomes.count("ok")
    assert [r.seq for r in records] == list(range(len(records)))


def test_every_written_record_is_intact_and_readable(tmp_path: Path) -> None:
    path = tmp_path / ".benchlock" / "ledger.jsonl"
    _run_writers(path, WRITERS)
    book = Ledger(path)
    runs = book.runs(StreamKind.SYSTEM)
    assert runs, "at least one writer should have succeeded"
    for run in runs:
        assert run.n == 40
        assert 0.0 <= run.mean <= 1.0


def test_no_line_is_ever_half_written(tmp_path: Path) -> None:
    """An interleaved write would leave a truncated line, which is unparseable JSON."""
    path = tmp_path / ".benchlock" / "ledger.jsonl"
    _run_writers(path, WRITERS)
    for line in path.read_text().splitlines():
        if line.strip():
            json.loads(line)  # raises if a record was interleaved mid-write


@pytest.mark.slow
def test_the_stress_test_a_hundred_times(tmp_path: Path) -> None:
    """The spec's requirement: run it 100 times, and never see a corrupt chain."""
    for repetition in range(100):
        path = tmp_path / f"run{repetition}" / "ledger.jsonl"
        outcomes = _run_writers(path, 8)
        assert not [o for o in outcomes if o.startswith("crashed")]
        records = Ledger(path).verify()
        assert [r.seq for r in records] == list(range(len(records))), (
            f"repetition {repetition} produced a broken chain"
        )


def test_a_lock_timeout_fails_cleanly_rather_than_interleaving(tmp_path: Path) -> None:
    """When the lock cannot be taken, the write is refused with an explanation."""
    import fcntl
    import os

    path = tmp_path / ".benchlock" / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    held = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        from benchlock.ledger.log import _file_lock

        with pytest.raises(LedgerError) as excinfo, _file_lock(path, timeout=0.2):
            pass  # pragma: no cover
        assert "could not lock" in excinfo.value.message
        assert "serialise your CI jobs" in excinfo.value.hint
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)
