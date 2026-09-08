"""Deterministic ground-truth stream generator.

Every stream is a pure function of its seed and its parameters, so a fixture regenerates
byte-identically and a benchmark cell can be re-run years later and produce the same
numbers. Nothing here reads a clock or an environment variable.

The model, stated plainly so the assumptions are visible:

* Each eval item has a latent quality. The judge maps quality to a score with
  item-level noise (its self-disagreement) plus, optionally, a run-level shared offset —
  the shape a provider-side change actually takes.
* A **judge shift** moves every score, on both the system suite and the anchor set,
  because it is a change in the measuring instrument.
* A **system shift** moves only the system suite, because the system under test does not
  touch the anchor items. That asymmetry is the entire identification argument, and here
  it is imposed by construction so that ground truth is known.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from benchlock.model.pins import (
    AnchorMode,
    AnchorPin,
    JudgePin,
    NoiseFloor,
    baseline_scores_hash,
)
from benchlock.model.streams import Observation, RunRecord, StreamKind, suite_hash_of

ScoreType = Literal["binary", "likert5", "continuous"]


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """Everything that determines a pair of streams, plus its ground truth."""

    name: str
    seed: int
    n_runs: int = 60
    #: Runs before the change point; the change begins at this index.
    change_at: int = 20
    system_items: int = 200
    anchor_items: int = 200
    #: Judge self-disagreement on a single item, on the normalised [0,1] scale.
    per_item_sd: float = 0.08
    #: Run-level offset shared by every item, the shape a provider-side change takes.
    shared_sd: float = 0.0
    #: Ground truth. A judge shift moves both streams; a system shift moves only the system.
    judge_shift: float = 0.0
    system_shift: float = 0.0
    system_level: float = 0.75
    anchor_level: float = 0.70
    score_type: ScoreType = "continuous"
    #: Replicate scorings of the anchor set at baseline, for the noise floor.
    replicates: int = 5
    #: Baseline runs of the system stream, before monitoring starts.
    baseline_runs: int = 8
    judge_model: str = "claude-sonnet-4-5-20250929"
    #: Model string after the change, when the provider rotated the snapshot.
    judge_model_after: str | None = None
    scale: tuple[float, float] = (1.0, 5.0)

    @property
    def ground_truth(self) -> str:
        if self.judge_shift and self.system_shift:
            return "both"
        if self.judge_shift:
            return "judge"
        if self.system_shift:
            return "system"
        return "stable"


def _quantise(values: np.ndarray, score_type: ScoreType) -> np.ndarray:
    """Map continuous latent scores onto the rubric's actual output granularity."""
    clipped = np.clip(values, 0.0, 1.0)
    if score_type == "binary":
        return (clipped >= 0.5).astype(float)
    if score_type == "likert5":
        return np.round(clipped * 4.0) / 4.0
    return clipped


def _judge_pin(model: str, scale: tuple[float, float]) -> JudgePin:
    """A pin identical in shape to one the shipped CLI would write.

    The params come from the adapter rather than being spelled out here. A fixture that
    hard-codes a different set produces streams whose pins no real run could reproduce,
    which turns any test comparing a seeded ledger against a live config into a test of
    whether two hard-coded dicts happen to match.
    """
    from benchlock.judge.anthropic import AnthropicJudge

    return JudgePin.build(
        provider="anthropic",
        model=model,
        rubric_text="Score the answer 1-5 for helpfulness and factual accuracy.\n",
        params=AnthropicJudge().params,
        scale=scale,
    )


def _observations(
    rng: np.random.Generator,
    level: float,
    shift: float,
    n_items: int,
    spec: StreamSpec,
    prefix: str,
) -> tuple[Observation, ...]:
    shared = rng.normal(0.0, spec.shared_sd) if spec.shared_sd > 0 else 0.0
    latent = level + shift + shared + rng.normal(0.0, spec.per_item_sd, n_items)
    scores = _quantise(latent, spec.score_type)
    lo, hi = spec.scale
    return tuple(
        Observation(
            item_id=f"{prefix}-{i}",
            score=float(s),
            raw_score=lo + float(s) * (hi - lo),
            scale=spec.scale,
        )
        for i, s in enumerate(scores)
    )


def generate(spec: StreamSpec) -> tuple[list[RunRecord], list[RunRecord]]:
    """Build (system runs, anchor runs) for one spec. Deterministic in `spec.seed`."""
    rng = np.random.default_rng(spec.seed)

    # The anchor's noise floor is what a real `benchlock baseline` would have measured, so
    # it is stated here from the same parameters the generator uses.
    run_mean_sd = float(np.sqrt(spec.shared_sd**2 + spec.per_item_sd**2 / spec.anchor_items))
    noise_floor = NoiseFloor(
        per_item_sd=spec.per_item_sd,
        run_mean_sd=run_mean_sd,
        replicates=spec.replicates,
        n_items=spec.anchor_items,
    )
    anchor_ids = [f"anchor-{i}" for i in range(spec.anchor_items)]
    anchor_pin = AnchorPin(
        mode=AnchorMode.FROZEN_SELF,
        item_set_hash=suite_hash_of(anchor_ids),
        baseline_scores_hash=baseline_scores_hash(dict.fromkeys(anchor_ids, spec.anchor_level)),
        n=spec.anchor_items,
        noise_floor=noise_floor,
    )

    before = _judge_pin(spec.judge_model, spec.scale)
    after = _judge_pin(spec.judge_model_after or spec.judge_model, spec.scale)

    system_runs: list[RunRecord] = []
    anchor_runs: list[RunRecord] = []
    for t in range(spec.n_runs):
        changed = t >= spec.change_at
        judge_shift = spec.judge_shift if changed else 0.0
        system_shift = spec.system_shift if changed else 0.0
        pin = after if changed else before

        system_obs = _observations(
            rng, spec.system_level, judge_shift + system_shift, spec.system_items, spec, "item"
        )
        system_runs.append(
            RunRecord(
                run_id=f"{spec.name}-sys-{t:04d}",
                run_index=t,
                kind=StreamKind.SYSTEM,
                observations=system_obs,
                suite_hash=suite_hash_of(o.item_id for o in system_obs),
                judge_pin=pin,
                anchor_pin=None,
            )
        )
        # The judge moves the anchor set too; the system does not touch it.
        anchor_obs = _observations(
            rng, spec.anchor_level, judge_shift, spec.anchor_items, spec, "anchor"
        )
        anchor_runs.append(
            RunRecord(
                run_id=f"{spec.name}-anc-{t:04d}",
                run_index=t,
                kind=StreamKind.ANCHOR,
                observations=anchor_obs,
                suite_hash=suite_hash_of(o.item_id for o in anchor_obs),
                judge_pin=pin,
                anchor_pin=anchor_pin,
            )
        )
    return system_runs, anchor_runs


# ---------------------------------------------------------------------------------------
# Serialisation, so fixtures are committed rather than regenerated at test time
# ---------------------------------------------------------------------------------------


def _run_to_json(run: RunRecord) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "run_index": run.run_index,
        "kind": run.kind.value,
        "suite_hash": run.suite_hash,
        "judge_pin": run.judge_pin.to_json(),
        "anchor_pin": run.anchor_pin.to_json() if run.anchor_pin else None,
        "epoch": run.epoch,
        "observations": [
            {"item_id": o.item_id, "score": o.score, "raw": o.raw_score} for o in run.observations
        ],
    }


def _run_from_json(data: dict[str, Any]) -> RunRecord:
    judge_pin = JudgePin.from_json(data["judge_pin"])
    return RunRecord(
        run_id=data["run_id"],
        run_index=int(data["run_index"]),
        kind=StreamKind(data["kind"]),
        observations=tuple(
            Observation(
                item_id=o["item_id"],
                score=float(o["score"]),
                raw_score=float(o["raw"]),
                scale=judge_pin.scale,
            )
            for o in data["observations"]
        ),
        suite_hash=data["suite_hash"],
        judge_pin=judge_pin,
        anchor_pin=AnchorPin.from_json(data["anchor_pin"]) if data["anchor_pin"] else None,
        epoch=int(data.get("epoch", 0)),
    )


def digest(system_runs: Sequence[RunRecord], anchor_runs: Sequence[RunRecord]) -> str:
    """A compact fingerprint of a generated stream pair.

    Fixtures are stored as *specs plus this digest* rather than as materialised runs: the
    full per-item observations for the golden set come to 26 MB, which has no business in
    a git history when the generator is deterministic and committed beside it. The digest
    is what turns "deterministic" from a claim into a test — if numpy's Generator ever
    changed its stream, every fixture would fail loudly instead of silently shifting the
    ground truth underneath the golden verdicts.
    """
    payload = [
        [round(r.mean, 12) for r in system_runs],
        [round(r.mean, 12) for r in anchor_runs],
        [round(o.score, 12) for o in system_runs[0].observations[:8]] if system_runs else [],
        [round(o.score, 12) for o in anchor_runs[0].observations[:8]] if anchor_runs else [],
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def spec_to_json(spec: StreamSpec) -> dict[str, Any]:
    data = asdict(spec)
    data["scale"] = list(spec.scale)
    return data


def spec_from_json(data: dict[str, Any]) -> StreamSpec:
    fields = dict(data)
    fields["scale"] = tuple(fields["scale"])
    fields.pop("ground_truth", None)
    fields.pop("digest", None)
    return StreamSpec(**fields)


def write_manifest(specs: Sequence[StreamSpec], path: Path) -> None:
    """Write specs + digests. Regenerating must leave this file byte-identical."""
    entries = []
    for spec in specs:
        system_runs, anchor_runs = generate(spec)
        entries.append(
            {
                **spec_to_json(spec),
                "ground_truth": spec.ground_truth,
                "digest": digest(system_runs, anchor_runs),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"streams": entries}, indent=2, sort_keys=True) + "\n")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text())["streams"])
