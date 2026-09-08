"""Regressions for the defects the post-release audit found.

Each test here corresponds to something that was broken in a shipped commit and that no
existing test caught. They are grouped by what made them invisible, because that is the
more useful lesson: every one of them lived in the gap between what the benchmarks
exercise (the engine, fed hand-built streams) and what a user runs (the CLI, building a
ledger of its own).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchlock.attribute.engine import decide
from benchlock.config import AttributionConfig, BenchlockConfig, JudgeParams, ProviderKind
from benchlock.model.streams import StreamKind

runner = CliRunner()

CONFIG = """\
version: 1
alpha: 0.05
score_scale: [1, 5]
min_runs: 8
min_obs: 3
system:
  adapter: {adapter}
  path: ./results.jsonl
anchor:
  mode: frozen-self
  n: 4
  cadence: {cadence}
  noise_replicates: 3
  seed: 0
judge:
  provider: anthropic
  model: claude-sonnet-5
  rubric: ./rubric.md
gate:
  fail_on: [system, both]
"""


def project(tmp_path: Path, *, adapter: str = "jsonl", cadence: int = 1) -> Path:
    (tmp_path / "rubric.md").write_text("Score the answer 1-5. Reply with a bare number.\n")
    (tmp_path / "benchlock.yaml").write_text(CONFIG.format(adapter=adapter, cadence=cadence))
    (tmp_path / "anchors.jsonl").write_text(
        "\n".join(
            json.dumps({"item_id": f"i{i}", "prompt_input": f"q{i}", "output": f"a{i}"})
            for i in range(4)
        )
        + "\n"
    )
    (tmp_path / "results.jsonl").write_text(
        "\n".join(json.dumps({"item_id": f"s{i}", "score": 4.0}) for i in range(4)) + "\n"
    )
    return tmp_path


def cli(root: Path, *args: str):
    return runner.invoke(
        app,
        [*args, "--config", str(root / "benchlock.yaml"), "--ledger", str(root / "led.jsonl")],
    )


from benchlock.cli import app  # noqa: E402 - imported after the helpers that use it

# --- the CLI could not reach a real judge at all -----------------------------------------


def test_a_real_provider_resolves_to_a_real_adapter() -> None:
    """`_judge_adapter` used to `_die` for every provider, so no hosted judge was reachable.

    The whole product is "re-score a frozen anchor set with YOUR judge every run". A CLI
    that can only ever drive the simulated judge measures the stability of a seeded RNG.
    """
    from benchlock.cli import _judge_adapter

    cfg = BenchlockConfig()
    assert cfg.judge.provider is ProviderKind.ANTHROPIC
    adapter = _judge_adapter(cfg, simulate=True)
    assert adapter.pin().model.startswith("simulated::")

    # The real path must build a provider adapter rather than exiting. It is constructed
    # without a key and never called, so this touches no network.
    from benchlock.judge.anthropic import AnthropicJudge

    judge = AnthropicJudge(model=cfg.judge.model, rubric_text="r", scale=cfg.score_scale)
    assert judge.describe().provider == "anthropic"
    assert "temperature" not in judge.params, (
        "current models reject temperature with a 400; the adapter default must not send it"
    )


def test_judge_params_default_to_unset_so_the_adapter_default_applies() -> None:
    """The schema used to force `temperature: 0.0` on every call.

    Current Anthropic models reject it, so a config written by `benchlock init` could not
    talk to the model the README advertises.
    """
    assert JudgeParams().model_dump(exclude_none=True) == {}
    # Provider-specific keys must survive: the valid set belongs to the provider.
    params = JudgeParams(**{"max_tokens": 16, "thinking": {"type": "disabled"}})
    dumped = params.model_dump(exclude_none=True)
    assert dumped["thinking"] == {"type": "disabled"}


def test_the_judge_pin_comes_from_the_adapter_that_does_the_scoring(tmp_path: Path) -> None:
    """Rebuilding the pin beside the adapter wrote a params_hash no run could reproduce."""
    from benchlock.cli import _judge_adapter, _judge_pin

    root = project(tmp_path)
    cfg = BenchlockConfig.load(root / "benchlock.yaml")
    assert _judge_pin(cfg, simulate=True) == _judge_adapter(cfg, simulate=True).pin()


# --- the two streams were paired by a counter that did not line up ------------------------


def test_an_anchor_rescore_carries_the_index_of_the_system_run_it_accompanies(
    tmp_path: Path,
) -> None:
    """The difference-in-differences legs join on `run_index`.

    Anchor runs used to be numbered by a per-stream counter that already contained the K
    noise-floor replicates, so system run j was compared against the anchor scoring taken
    K runs earlier. Every ledger the CLI wrote was misaligned; no benchmark caught it
    because the simulation builds both streams with matching indices.
    """
    from benchlock.ledger.log import Ledger

    root = project(tmp_path)
    baseline = cli(root, "baseline", "--anchors", str(root / "anchors.jsonl"), "--simulate")
    assert baseline.exit_code == 0, baseline.output
    for _ in range(3):
        cli(root, "observe", str(root / "results.jsonl"), "--simulate", "--rescore-anchors")

    book = Ledger(root / "led.jsonl")
    system = [r.run_index for r in book.runs(StreamKind.SYSTEM)]
    anchor = [r.run_index for r in book.runs(StreamKind.ANCHOR)]
    replicates = 3
    assert system == [0, 1, 2]
    assert anchor[:replicates] == [0, 1, 2], "the K replicates come first, in ledger order"
    assert anchor[replicates:] == system, (
        f"each re-score must carry its system run's index; got {anchor[replicates:]} for {system}"
    )


def test_cadence_is_enforced_not_merely_validated(tmp_path: Path) -> None:
    """`anchor.cadence` was printed by `plan` and provisioned against, but never obeyed."""
    from benchlock.ledger.log import Ledger

    root = project(tmp_path, cadence=3)
    cli(root, "baseline", "--anchors", str(root / "anchors.jsonl"), "--simulate")
    before = len(Ledger(root / "led.jsonl").runs(StreamKind.ANCHOR))
    for _ in range(3):
        cli(root, "observe", str(root / "results.jsonl"), "--simulate", "--rescore-anchors")
    after = len(Ledger(root / "led.jsonl").runs(StreamKind.ANCHOR))
    assert after - before == 1, "with cadence 3, three system runs must re-score anchors once"


# --- input the tool itself invites used to produce tracebacks -----------------------------


def test_a_directory_as_the_results_path_is_refused_with_a_sentence(tmp_path: Path) -> None:
    """`benchlock init` writes a DIRECTORY as `system.path` when it detects no framework."""
    from benchlock.adapters.jsonl import IngestError
    from benchlock.adapters.jsonl import load as load_jsonl

    (tmp_path / "results").mkdir()
    with pytest.raises(IngestError) as excinfo:
        load_jsonl(tmp_path / "results", (1.0, 5.0))
    assert "directory" in str(excinfo.value)


def test_a_binary_file_is_refused_with_a_sentence(tmp_path: Path) -> None:
    """The Inspect AI `.eval` archive that `init` detects is a binary file."""
    from benchlock.adapters.jsonl import IngestError
    from benchlock.adapters.jsonl import load as load_jsonl

    path = tmp_path / "log.eval"
    path.write_bytes(b"PK\x03\x04\xd7\xd8\xd9 not utf-8")
    with pytest.raises(IngestError) as excinfo:
        load_jsonl(path, (1.0, 5.0))
    assert "UTF-8" in str(excinfo.value)


def test_a_truncated_anchor_store_names_the_file_and_the_line(tmp_path: Path) -> None:
    """An interrupted `baseline` leaves one; the crash used to land AFTER a ledger append."""
    from benchlock.anchor.modes import AnchorModeError, load_anchors

    path = tmp_path / "anchors.jsonl"
    path.write_text('{"item_id": "a"}\n{"item_id": "b", "prom\n')
    with pytest.raises(AnchorModeError) as excinfo:
        load_anchors(path)
    assert "line 2" in excinfo.value.message


# --- a verdict has to be reproducible from what was recorded ------------------------------


def test_target_shift_is_recorded_because_it_changes_the_verdict() -> None:
    """`replay` used to re-derive under its own flag and call the difference a regression."""
    from bench.sim.generate import StreamSpec, generate

    spec = StreamSpec(
        name="r", seed=5, n_runs=30, change_at=12, per_item_sd=0.08, system_shift=-0.10
    )
    system, anchor = generate(spec)
    tight = decide(list(system), list(anchor), AttributionConfig(target_shift=0.05))
    assert tight.target_shift == 0.05
    assert tight.to_json()["target_shift"] == 0.05
    wide = decide(list(system), list(anchor), AttributionConfig(target_shift=0.20))
    assert wide.target_shift == 0.20


def test_a_shift_past_the_estimation_band_is_flagged_as_a_lower_bound() -> None:
    """The flag was computed and then dropped, so the reported interval silently excluded
    the true shift."""
    from bench.sim.generate import StreamSpec, generate

    spec = StreamSpec(
        name="big", seed=3, n_runs=40, change_at=12, per_item_sd=0.08, system_shift=-0.30
    )
    system, anchor = generate(spec)
    attribution = decide(list(system), list(anchor), AttributionConfig(target_shift=0.05))
    assert attribution.evidence.system_shift_saturated, (
        "a -0.30 move against a 0.20 band must be reported as saturated"
    )
    assert attribution.evidence.to_json()["system_shift_saturated"] is True
    assert attribution.evidence.system_shift.lower > -0.30, (
        "this test is only meaningful while the interval really does exclude the truth"
    )
    # And the renderer has to say so, not just carry the flag.
    from benchlock.report.human import render_verdict_block

    assert "LOWER BOUND" in render_verdict_block(attribution)
