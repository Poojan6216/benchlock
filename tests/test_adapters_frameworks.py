"""Phase 8.1 verify: each framework adapter round-trips, or fails loudly.

**About these fixtures.** promptfoo, Inspect AI and DeepEval are not installed in this
build environment, so the fixtures are constructed from each project's *documented* output
shape rather than captured from a real run. That is a real weakness and is stated plainly
here rather than left for a reader to discover: a schema that has drifted from its docs
would pass these tests and fail in the field.

What the tests do guarantee is the property that matters most for a measurement tool: an
unrecognised shape is **refused with a message naming what was actually found**, never
guessed at. A wrong field silently produces a stream of plausible numbers that mean
something else, and every verdict built on top of it would be confidently wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchlock.adapters import deepeval, inspect_ai, promptfoo
from benchlock.adapters.jsonl import IngestError

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "frameworks"


# --- promptfoo ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["promptfoo-v3.json", "promptfoo-v2.json"])
def test_promptfoo_shapes_round_trip(name: str) -> None:
    observations = promptfoo.load(FIXTURES / name, (0.0, 1.0))
    assert len(observations) >= 35
    assert all(0.0 <= o.score <= 1.0 for o in observations)
    assert all(o.item_id.startswith("case-") for o in observations)
    # Item ids are stable across runs, which is what the anchor set depends on.
    assert len({o.item_id for o in observations}) == len(observations)


def test_promptfoo_refuses_an_unrecognised_shape(tmp_path: Path) -> None:
    path = tmp_path / "not-promptfoo.json"
    path.write_text('{"summary": {"passed": 10}, "meta": {}}')
    with pytest.raises(IngestError) as excinfo:
        promptfoo.load(path, (0.0, 1.0))
    rendered = excinfo.value.render()
    assert "no results array found" in rendered
    assert "meta, summary" in rendered, "the error must name what it actually saw"
    assert "refuses to guess" in rendered


def test_promptfoo_refuses_ambiguous_named_scores() -> None:
    """Two metrics and no overall score is a question for the user, not the adapter."""
    with pytest.raises(IngestError) as excinfo:
        promptfoo.load(FIXTURES / "promptfoo-ambiguous.json", (0.0, 1.0))
    assert "2 named scores" in excinfo.value.render()
    assert "one benchlock config per metric" in excinfo.value.render()


def test_promptfoo_missing_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="promptfoo eval"):
        promptfoo.load(tmp_path / "nope.json", (0.0, 1.0))


def test_promptfoo_scale_detection_is_a_suggestion() -> None:
    assert promptfoo.detect_scale(FIXTURES / "promptfoo-v3.json") == (0.0, 1.0)
    assert promptfoo.detect_scale(FIXTURES / "promptfoo-ambiguous.json") is None


# --- Inspect AI ----------------------------------------------------------------------------


def test_inspect_json_log_with_letter_grades() -> None:
    observations = inspect_ai.load(FIXTURES / "inspect-log.json", (0.0, 1.0))
    assert len(observations) == 40
    # "C" -> 1.0 and "I" -> 0.0, mapped explicitly rather than coerced.
    assert set(o.score for o in observations) == {0.0, 1.0}


def test_inspect_eval_archive_is_read() -> None:
    observations = inspect_ai.load(FIXTURES / "inspect-log.eval", (0.0, 1.0))
    assert len(observations) == 40
    assert all(0.0 <= o.score <= 1.0 for o in observations)


def test_inspect_refuses_an_unrecognised_grade(tmp_path: Path) -> None:
    path = tmp_path / "odd.json"
    path.write_text('{"samples": [{"id": "s0", "scores": {"g": {"value": "EXCELLENT"}}}]}')
    with pytest.raises(IngestError) as excinfo:
        inspect_ai.load(path, (0.0, 1.0))
    rendered = excinfo.value.render()
    assert "unrecognised grade 'EXCELLENT'" in rendered
    assert "will not invent a value" in rendered


def test_inspect_refuses_multiple_scorers_without_a_choice() -> None:
    with pytest.raises(IngestError) as excinfo:
        inspect_ai.load(FIXTURES / "inspect-two-scorers.json", (0.0, 1.0))
    assert "2 scorers" in excinfo.value.render()


def test_inspect_accepts_a_named_scorer() -> None:
    observations = inspect_ai.load(FIXTURES / "inspect-two-scorers.json", (0.0, 1.0), scorer="a")
    assert observations[0].score == 1.0


def test_inspect_refuses_a_non_inspect_file(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text('{"runs": [], "meta": 1}')
    with pytest.raises(IngestError) as excinfo:
        inspect_ai.load(path, (0.0, 1.0))
    assert "no `samples` array" in excinfo.value.render()
    assert "meta, runs" in excinfo.value.render()


# --- DeepEval -------------------------------------------------------------------------------


def test_deepeval_run_round_trips() -> None:
    observations = deepeval.load(FIXTURES / "deepeval-run.json", (0.0, 1.0))
    assert len(observations) == 40
    assert all(o.item_id.startswith("t") for o in observations)
    assert all(0.0 <= o.score <= 1.0 for o in observations)


def test_deepeval_refuses_ambiguous_metrics() -> None:
    """Averaging metrics is a modelling decision, not an ingest decision."""
    with pytest.raises(IngestError) as excinfo:
        deepeval.load(FIXTURES / "deepeval-two-metrics.json", (0.0, 1.0))
    rendered = excinfo.value.render()
    assert "2 metrics" in rendered
    assert "belongs to you, not to an ingest adapter" in rendered


def test_deepeval_accepts_a_named_metric() -> None:
    observations = deepeval.load(
        FIXTURES / "deepeval-two-metrics.json", (0.0, 1.0), metric="Faithfulness"
    )
    assert observations[0].score == pytest.approx(0.3)


def test_deepeval_refuses_an_unrecognised_shape(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text('{"suite": "x", "outcome": "pass"}')
    with pytest.raises(IngestError) as excinfo:
        deepeval.load(path, (0.0, 1.0))
    assert "no test-case array found" in excinfo.value.render()
    assert "outcome, suite" in excinfo.value.render()


# --- the property that matters across all three ------------------------------------------------


@pytest.mark.parametrize(
    "adapter", [promptfoo, inspect_ai, deepeval], ids=["promptfoo", "inspect_ai", "deepeval"]
)
def test_every_adapter_refuses_rather_than_guesses(adapter, tmp_path: Path) -> None:
    """The shared contract: an unrecognised schema is an error, never a best effort."""
    path = tmp_path / "mystery.json"
    path.write_text('{"totally": "unrelated", "shape": [1, 2, 3]}')
    with pytest.raises(IngestError) as excinfo:
        adapter.load(path, (0.0, 1.0))
    rendered = excinfo.value.render()
    assert "jsonl" in rendered.lower() or "universal" in rendered.lower(), (
        "the error should point at the universal adapter as the escape hatch"
    )


@pytest.mark.parametrize(
    "adapter", [promptfoo, inspect_ai, deepeval], ids=["promptfoo", "inspect_ai", "deepeval"]
)
def test_every_adapter_normalises_through_the_same_contract(adapter, tmp_path: Path) -> None:
    """All three end up as Observations on [0,1] with their raw value retained."""
    fixtures = {
        promptfoo: "promptfoo-v3.json",
        inspect_ai: "inspect-log.json",
        deepeval: "deepeval-run.json",
    }
    observations = adapter.load(FIXTURES / fixtures[adapter], (0.0, 1.0))
    assert observations
    for o in observations:
        assert 0.0 <= o.score <= 1.0
        assert o.scale == (0.0, 1.0)
        assert o.item_id
