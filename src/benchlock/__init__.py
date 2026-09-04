"""Benchlock: anytime-valid attribution of eval-score movement to judge or system."""

from __future__ import annotations

__version__ = "0.1.0"

#: Bumped whenever the verdict lattice or the statistics that feed it change.
#: Recorded in every ledger record so `benchlock replay` can detect a semantics
#: change instead of silently rewriting history (Phase 4.2).
DECISION_SEMANTICS_VERSION = 1

__all__ = ["DECISION_SEMANTICS_VERSION", "__version__"]
