"""`benchlock replay`: re-derive every historical verdict and check it still holds.

This is the property that makes a verdict auditable rather than merely remembered. Given
the ledger and the config, every decision the tool ever issued can be recomputed from its
inputs and compared to what was actually reported at the time. A mismatch is a build
failure, not a warning.

Three things it catches, and they are different failures:

* **A tampered ledger.** The hash chain catches edits directly, before replay begins.
* **A behaviour change nobody meant to make.** Refactor the lattice, get a different
  verdict on run 47 of a stream from March, and replay names run 47.
* **A behaviour change somebody did mean to make.** Bump `DECISION_SEMANTICS_VERSION` and
  replay reports *both* verdicts and lists the divergences, rather than quietly rewriting
  history to agree with the new code. That distinction is the whole point of versioning
  the semantics: old verdicts were not wrong, they were issued under different rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import benchlock
from benchlock.attribute.engine import decide
from benchlock.config import AttributionConfig
from benchlock.ledger.log import Ledger, RecordType
from benchlock.model.streams import RunRecord, StreamKind
from benchlock.model.verdict import AttributionRefusedError


@dataclass(frozen=True, slots=True)
class Divergence:
    """One historical verdict that today's code does not reproduce."""

    seq: int
    at_system_run: int
    recorded_verdict: str
    recorded_rule: str
    replayed_verdict: str
    replayed_rule: str
    recorded_semantics: int
    replayed_semantics: int

    @property
    def semantics_changed(self) -> bool:
        return self.recorded_semantics != self.replayed_semantics

    def describe(self) -> str:
        kind = (
            f"decision semantics v{self.recorded_semantics} -> v{self.replayed_semantics}"
            if self.semantics_changed
            else "SAME semantics version, so this is a regression"
        )
        return (
            f"record {self.seq} (system run {self.at_system_run}): "
            f"recorded {self.recorded_verdict}/{self.recorded_rule}, "
            f"replayed {self.replayed_verdict}/{self.replayed_rule}  [{kind}]"
        )


@dataclass(frozen=True, slots=True)
class ReplayReport:
    checked: int
    divergences: tuple[Divergence, ...] = field(default_factory=tuple)
    refusals: tuple[str, ...] = field(default_factory=tuple)
    #: Read at call time rather than bound at import, so a semantics bump is visible to a
    #: replay running in the same process (which is exactly how the tests exercise it).
    semantics_version: int = 0

    @property
    def ok(self) -> bool:
        return not self.divergences and not self.refusals

    @property
    def first_divergence(self) -> Divergence | None:
        return self.divergences[0] if self.divergences else None

    def to_json(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "ok": self.ok,
            "semantics_version": self.semantics_version,
            "divergences": [
                {
                    "seq": d.seq,
                    "at_system_run": d.at_system_run,
                    "recorded": {"verdict": d.recorded_verdict, "rule_id": d.recorded_rule},
                    "replayed": {"verdict": d.replayed_verdict, "rule_id": d.replayed_rule},
                    "recorded_semantics": d.recorded_semantics,
                    "replayed_semantics": d.replayed_semantics,
                    "semantics_changed": d.semantics_changed,
                }
                for d in self.divergences
            ],
            "refusals": list(self.refusals),
        }


def _prefix(runs: Sequence[RunRecord], upto_index: int) -> list[RunRecord]:
    """Runs up to and including `upto_index`, which is how the stream looked at the time."""
    return [r for r in runs if r.run_index <= upto_index]


def replay(ledger: Ledger, config: AttributionConfig) -> ReplayReport:
    """Re-derive every recorded verdict from its own prefix of the ledger."""
    records = ledger.verify()  # the chain is checked before anything else
    system_all = ledger.runs(StreamKind.SYSTEM)
    anchor_all = ledger.runs(StreamKind.ANCHOR)

    current_semantics = benchlock.DECISION_SEMANTICS_VERSION
    divergences: list[Divergence] = []
    refusals: list[str] = []
    checked = 0

    for record in records:
        if record.type is not RecordType.VERDICT:
            continue
        checked += 1
        payload = record.payload
        recorded = dict(payload["attribution"])
        at_system = int(payload["at_system_run"])
        at_anchor = int(payload["at_anchor_run"])

        try:
            replayed = decide(
                _prefix(system_all, at_system), _prefix(anchor_all, at_anchor), config
            )
        except AttributionRefusedError as exc:
            refusals.append(
                f"record {record.seq} (system run {at_system}): replay refused to decide "
                f"({exc.rule_id}), but a verdict of `{recorded['verdict']}` was recorded"
            )
            continue

        if replayed.verdict.value != recorded["verdict"] or replayed.rule_id != recorded["rule_id"]:
            divergences.append(
                Divergence(
                    seq=record.seq,
                    at_system_run=at_system,
                    recorded_verdict=str(recorded["verdict"]),
                    recorded_rule=str(recorded["rule_id"]),
                    replayed_verdict=replayed.verdict.value,
                    replayed_rule=replayed.rule_id,
                    recorded_semantics=record.semantics_version,
                    replayed_semantics=current_semantics,
                )
            )

    return ReplayReport(
        checked=checked,
        divergences=tuple(divergences),
        refusals=tuple(refusals),
        semantics_version=current_semantics,
    )
