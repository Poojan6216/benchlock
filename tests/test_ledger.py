"""Phase 0.4 verify: the chain is tamper-evident, and no eval content reaches the file."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from benchlock.ledger.log import (
    GENESIS_HASH,
    RUN_PAYLOAD_KEYS,
    Ledger,
    LedgerError,
    RecordType,
    hash_item_id,
    new_run_id,
)
from benchlock.model.streams import StreamKind
from tests.conftest import make_run


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / ".benchlock" / "ledger.jsonl")


def _seed(ledger: Ledger, judge_pin, anchor_pin, n: int = 5) -> None:
    for i in range(n):
        ledger.append_run(
            make_run(
                run_index=i,
                kind=StreamKind.SYSTEM,
                scores=[0.8, 0.7, 0.9],
                judge_pin=judge_pin,
            )
        )
        ledger.append_run(
            make_run(
                run_index=i,
                kind=StreamKind.ANCHOR,
                scores=[0.7, 0.7, 0.75],
                judge_pin=judge_pin,
                anchor_pin=anchor_pin,
                prefix="anchor",
            )
        )


# --- chain integrity ---------------------------------------------------------------------


def test_empty_ledger_verifies_and_starts_at_genesis(ledger: Ledger) -> None:
    assert ledger.verify() == []
    assert ledger.head() == (0, GENESIS_HASH)


def test_appends_form_a_chain(ledger: Ledger, judge_pin, anchor_pin) -> None:
    _seed(ledger, judge_pin, anchor_pin, n=3)
    records = ledger.verify()
    assert [r.seq for r in records] == list(range(6))
    assert records[0].prev == GENESIS_HASH
    for earlier, later in itertools.pairwise(records):
        assert later.prev == earlier.hash


@pytest.mark.mandatory
def test_flipping_one_byte_anywhere_breaks_the_chain(ledger: Ledger, judge_pin, anchor_pin) -> None:
    _seed(ledger, judge_pin, anchor_pin, n=4)
    original = ledger.path.read_text()
    lines = original.splitlines()
    assert len(lines) == 8

    # Tamper with each record in turn: a changed score must always be detected,
    # and the reported index must be the record that was edited.
    for target in range(len(lines)):
        data = json.loads(lines[target])
        data["payload"]["observations"][0]["score"] = 0.123456
        tampered = list(lines)
        tampered[target] = json.dumps(data, sort_keys=True, separators=(",", ":"))
        ledger.path.write_text("\n".join(tampered) + "\n")

        with pytest.raises(LedgerError) as excinfo:
            ledger.verify()
        assert excinfo.value.index == target, (
            f"editing record {target} was reported at {excinfo.value.index}"
        )
    ledger.path.write_text(original)
    assert len(ledger.verify()) == 8


@pytest.mark.mandatory
def test_single_character_edit_is_detected(ledger: Ledger, judge_pin, anchor_pin) -> None:
    _seed(ledger, judge_pin, anchor_pin, n=2)
    raw = ledger.path.read_text()
    # Flip one character in the middle of the file, wherever it lands.
    pos = len(raw) // 2
    flipped = raw[:pos] + ("0" if raw[pos] != "0" else "1") + raw[pos + 1 :]
    ledger.path.write_text(flipped)
    with pytest.raises(LedgerError):
        ledger.verify()


def test_deleting_a_record_is_detected(ledger: Ledger, judge_pin, anchor_pin) -> None:
    _seed(ledger, judge_pin, anchor_pin, n=3)
    lines = ledger.path.read_text().splitlines()
    del lines[2]
    ledger.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError) as excinfo:
        ledger.verify()
    assert excinfo.value.index == 2


def test_reordering_records_is_detected(ledger: Ledger, judge_pin, anchor_pin) -> None:
    _seed(ledger, judge_pin, anchor_pin, n=3)
    lines = ledger.path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    ledger.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError) as excinfo:
        ledger.verify()
    assert excinfo.value.index == 1


def test_truncation_still_verifies_but_loses_history(ledger: Ledger, judge_pin, anchor_pin) -> None:
    # An append-only file truncated at a record boundary is still a valid chain; that is
    # inherent to hash chaining and is documented rather than pretended away.
    _seed(ledger, judge_pin, anchor_pin, n=3)
    lines = ledger.path.read_text().splitlines()
    ledger.path.write_text("\n".join(lines[:4]) + "\n")
    assert len(ledger.verify()) == 4


def test_appending_after_a_break_still_reports_the_break(
    ledger: Ledger, judge_pin, anchor_pin
) -> None:
    _seed(ledger, judge_pin, anchor_pin, n=2)
    lines = ledger.path.read_text().splitlines()
    data = json.loads(lines[0])
    data["payload"]["epoch"] = 99
    lines[0] = json.dumps(data, sort_keys=True, separators=(",", ":"))
    ledger.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError):
        ledger.append_run(
            make_run(run_index=9, kind=StreamKind.SYSTEM, scores=[0.5], judge_pin=judge_pin)
        )


# --- Hard Rule 9: nothing but scores, hashes and metadata --------------------------------


def _load_secrets() -> list[tuple[str, str]]:
    from tests.conftest import FIXTURES

    data = json.loads((FIXTURES / "secrets" / "seeded.json").read_text())
    return [(s["name"], s["value"]) for s in data["secrets"]]


@pytest.mark.mandatory
def test_seeded_secrets_never_reach_the_ledger(ledger: Ledger, judge_pin, anchor_pin) -> None:
    """Hard Rule 9. This test is mandatory and must never be skipped or xfailed.

    Secrets are seeded everywhere eval content can plausibly carry them: into item ids,
    into model outputs and judge rationales attached to the run, and into free-text
    metadata. None may appear in the ledger.
    """
    secrets = _load_secrets()
    assert len(secrets) == 15, "the fixture must carry all 15 formats"

    # Item ids that *are* the secret — the realistic case, since eval suites routinely
    # key items on the prompt text or a customer identifier.
    run = make_run(
        run_index=0,
        kind=StreamKind.SYSTEM,
        scores=[0.5 + i * 0.01 for i in range(len(secrets))],
        judge_pin=judge_pin,
    )
    run = type(run)(
        run_id=new_run_id(),
        run_index=0,
        kind=StreamKind.SYSTEM,
        observations=tuple(
            type(o)(item_id=f"case::{value}", score=o.score, raw_score=o.raw_score, scale=o.scale)
            for o, (_, value) in zip(run.observations, secrets, strict=True)
        ),
        suite_hash=run.suite_hash,
        judge_pin=run.judge_pin,
        anchor_pin=None,
    )
    ledger.append_run(run)
    ledger.append_run(
        make_run(
            run_index=0,
            kind=StreamKind.ANCHOR,
            scores=[0.7, 0.7],
            judge_pin=judge_pin,
            anchor_pin=anchor_pin,
            prefix="anchor",
        )
    )

    written = ledger.path.read_bytes().decode("utf-8")
    for name, value in secrets:
        assert value not in written, f"{name} leaked into the ledger"
        # Also check a lowercased form, in case of case-normalising serialisation.
        assert value.lower() not in written.lower(), f"{name} leaked (case-insensitively)"

    # And the ledger is still useful: the scores survived.
    assert len(ledger.runs(StreamKind.SYSTEM)[0].observations) == len(secrets)


@pytest.mark.mandatory
def test_run_payload_holds_only_allowlisted_keys(ledger: Ledger, judge_pin) -> None:
    ledger.append_run(
        make_run(run_index=0, kind=StreamKind.SYSTEM, scores=[0.5, 0.6], judge_pin=judge_pin)
    )
    record = ledger.verify()[0]
    assert set(record.payload) == RUN_PAYLOAD_KEYS
    for obs in record.payload["observations"]:
        assert set(obs) == {"item", "score", "raw"}


def test_item_ids_are_stored_hashed(ledger: Ledger, judge_pin) -> None:
    ledger.append_run(
        make_run(run_index=0, kind=StreamKind.SYSTEM, scores=[0.5], judge_pin=judge_pin)
    )
    stored = ledger.verify()[0].payload["observations"][0]["item"]
    assert stored != "item-0"
    assert stored == hash_item_id("item-0")
    assert len(stored) == 32


def test_writing_an_unknown_field_is_refused(ledger: Ledger) -> None:
    from benchlock.ledger.log import _check_allowlist

    with pytest.raises(LedgerError, match="Hard Rule 9"):
        _check_allowlist({"run_id": "x", "judge_rationale": "the answer mentioned the CEO"})
    with pytest.raises(LedgerError, match="Hard Rule 9"):
        _check_allowlist({"observations": [{"item": "a", "score": 1.0, "raw": 5, "output": "hi"}]})


# --- reconstruction -----------------------------------------------------------------------


def test_runs_round_trip_through_the_ledger(ledger: Ledger, judge_pin, anchor_pin) -> None:
    _seed(ledger, judge_pin, anchor_pin, n=4)
    system = ledger.runs(StreamKind.SYSTEM)
    anchor = ledger.runs(StreamKind.ANCHOR)
    assert len(system) == len(anchor) == 4
    assert [r.run_index for r in system] == [0, 1, 2, 3]
    assert system[0].mean == pytest.approx(0.8)
    assert anchor[0].anchor_pin is not None
    assert anchor[0].judge_pin == judge_pin
    # Scores survive the round trip exactly.
    assert system[0].scores() == (0.8, 0.7, 0.9)


def test_baseline_and_rebaseline_records(ledger: Ledger, judge_pin, anchor_pin) -> None:
    ledger.append_baseline(judge_pin=judge_pin, anchor_pin=anchor_pin, note="first freeze")
    assert ledger.epoch() == 0
    ledger.append_rebaseline(
        reason="judge-version-change",
        epoch=1,
        judge_pin=judge_pin,
        anchor_pin=anchor_pin,
        pin_delta=("model",),
    )
    assert ledger.epoch() == 1
    records = ledger.verify()
    assert records[0].type is RecordType.BASELINE
    assert records[1].type is RecordType.REBASELINE
    assert records[1].payload["reason"] == "judge-version-change"


def test_rebaseline_requires_a_reason(ledger: Ledger, judge_pin, anchor_pin) -> None:
    with pytest.raises(LedgerError, match="requires a reason"):
        ledger.append_rebaseline(reason="   ", epoch=1, judge_pin=judge_pin, anchor_pin=anchor_pin)


def test_ledger_is_append_only_on_disk(ledger: Ledger, judge_pin) -> None:
    ledger.append_run(
        make_run(run_index=0, kind=StreamKind.SYSTEM, scores=[0.5], judge_pin=judge_pin)
    )
    first = ledger.path.read_text()
    ledger.append_run(
        make_run(run_index=1, kind=StreamKind.SYSTEM, scores=[0.6], judge_pin=judge_pin)
    )
    second = ledger.path.read_text()
    assert second.startswith(first), "an append rewrote earlier bytes"
