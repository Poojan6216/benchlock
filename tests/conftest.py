"""Shared fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# The `confseq` oracle (0.0.11) predates NumPy 2.0, which removed the `np.float_` alias it
# uses in type annotations. Restoring the alias is a one-line shim and is strictly better
# than pinning the whole project to numpy<2 for the sake of a dev-only test oracle.
# Applied at conftest import so it is in place before pytest collects any test module.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

from benchlock.model.pins import (
    AnchorMode,
    AnchorPin,
    JudgePin,
    NoiseFloor,
    baseline_scores_hash,
)
from benchlock.model.streams import Observation, RunRecord, StreamKind, suite_hash_of

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def judge_pin() -> JudgePin:
    return JudgePin.build(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        rubric_text="Score the answer 1-5 for helpfulness.",
        params={"temperature": 0.0, "max_tokens": 512},
        scale=(1.0, 5.0),
    )


@pytest.fixture
def noise_floor() -> NoiseFloor:
    return NoiseFloor(per_item_sd=0.08, run_mean_sd=0.006, replicates=5, n_items=260)


@pytest.fixture
def anchor_pin(noise_floor: NoiseFloor) -> AnchorPin:
    scores = {f"anchor-{i}": 0.7 for i in range(260)}
    return AnchorPin(
        mode=AnchorMode.FROZEN_SELF,
        item_set_hash=suite_hash_of(scores),
        baseline_scores_hash=baseline_scores_hash(scores),
        n=len(scores),
        noise_floor=noise_floor,
    )


def make_run(
    *,
    run_index: int,
    kind: StreamKind,
    scores: list[float],
    judge_pin: JudgePin,
    anchor_pin: AnchorPin | None = None,
    scale: tuple[float, float] = (1.0, 5.0),
    prefix: str = "item",
    epoch: int = 0,
    run_id: str | None = None,
) -> RunRecord:
    """Build a RunRecord from already-normalised [0,1] scores."""
    obs = tuple(
        Observation(
            item_id=f"{prefix}-{i}",
            score=s,
            raw_score=scale[0] + s * (scale[1] - scale[0]),
            scale=scale,
        )
        for i, s in enumerate(scores)
    )
    return RunRecord(
        run_id=run_id or f"RUN{run_index:08d}",
        run_index=run_index,
        kind=kind,
        observations=obs,
        suite_hash=suite_hash_of(o.item_id for o in obs),
        judge_pin=judge_pin,
        anchor_pin=anchor_pin,
        epoch=epoch,
    )
