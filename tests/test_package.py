"""Smoke tests for the package skeleton (Phase 0.1)."""

from __future__ import annotations

import benchlock


def test_version_is_exposed() -> None:
    assert benchlock.__version__ == "0.1.0"


def test_decision_semantics_version_is_an_int() -> None:
    # Recorded in every ledger record; `benchlock replay` compares it (Phase 4.2).
    assert isinstance(benchlock.DECISION_SEMANTICS_VERSION, int)
    assert benchlock.DECISION_SEMANTICS_VERSION >= 1
