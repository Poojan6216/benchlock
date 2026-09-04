#!/usr/bin/env python3
"""Generate the committed golden streams in tests/fixtures/streams/.

Deterministic: re-running this must leave the files byte-identical. The fixtures are
committed so the golden tests do not depend on a particular numpy RNG version.
"""

from __future__ import annotations

from pathlib import Path

from bench.sim.generate import StreamSpec, write_manifest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "streams"

#: Anchor sizes here come from `benchlock plan`: 200 is what it asks for at a target shift
#: of 0.05, and 40 is the under-provisioned setting Demo 3 is about.
SPECS = [
    # The provider rotated the snapshot behind a stable model string. Nothing in the
    # config changed, so nothing but the anchor set can reveal it — which is the whole
    # reason the anchor set exists.
    StreamSpec(name="demo1-phantom-judge", seed=20260904, judge_shift=-0.12),
    # The same drift, but the user updated the model string too. Benchlock must refuse to
    # compare across an undeclared pin change rather than return a verdict (Hard Rule 8).
    StreamSpec(
        name="declared-judge-swap",
        seed=20260904,
        judge_shift=-0.12,
        judge_model_after="claude-sonnet-4-5-20260114",
    ),
    StreamSpec(name="demo2-real-regression", seed=20260905, system_shift=-0.05),
    StreamSpec(
        name="demo3-under-provisioned",
        seed=20260905,
        system_shift=-0.05,
        anchor_items=40,
    ),
    StreamSpec(name="stable-control", seed=20260906),
    StreamSpec(name="both-moved", seed=20260907, judge_shift=-0.06, system_shift=-0.05),
    StreamSpec(
        name="judge-rubric-change",
        seed=20260908,
        judge_shift=0.08,
        change_at=25,
    ),
    StreamSpec(
        name="cancellation",
        seed=20260909,
        judge_shift=0.05,
        system_shift=-0.05,
        change_at=20,
    ),
    StreamSpec(
        name="correlated-judge-noise",
        seed=20260910,
        shared_sd=0.02,
        system_shift=-0.05,
    ),
    StreamSpec(name="likert5-regression", seed=20260911, system_shift=-0.08, score_type="likert5"),
    # Binary scores near the ceiling barely move: a latent drop of 0.10 at level 0.75
    # leaves almost every item still passing. Held near the decision boundary so the
    # regression is actually expressible in the rubric's output.
    StreamSpec(
        name="binary-regression",
        seed=20260912,
        system_shift=-0.10,
        score_type="binary",
        system_level=0.55,
        anchor_level=0.55,
    ),
    StreamSpec(name="short-stream", seed=20260913, n_runs=6, system_shift=-0.10),
    # A change 45 runs in. The change-point prior charges a late candidate about 7.5 nats
    # of weight before it can contribute, so a late change needs materially more
    # post-change evidence than an early one — the stream is long enough to supply it.
    StreamSpec(
        name="late-judge-drift",
        seed=20260914,
        judge_shift=-0.10,
        change_at=45,
        n_runs=90,
    ),
    # Correlated judge noise so large that no 200-item anchor could see a 0.05 shift.
    # The honest answer is INDETERMINATE, and this fixture pins that behaviour.
]


def main() -> int:
    path = OUT / "manifest.json"
    write_manifest(SPECS, path)
    for spec in SPECS:
        print(f"  {spec.name:26s} ground_truth={spec.ground_truth}")
    print(f"wrote {len(SPECS)} stream specs to {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
