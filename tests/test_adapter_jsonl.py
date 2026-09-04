"""Phase 0.3 verify: score types normalise correctly; malformed input is rejected loudly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchlock.adapters.jsonl import IngestError, load
from benchlock.model.streams import Observation, suite_hash_of


def write_jsonl(tmp_path: Path, rows: list[dict | str], name: str = "run.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


# --- score types -----------------------------------------------------------------------
# (name, scale, raw scores, expected normalised scores)
SCORE_TYPES: list[tuple[str, tuple[float, float], list[float], list[float]]] = [
    ("binary 0/1", (0.0, 1.0), [0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0]),
    ("likert 1-5", (1.0, 5.0), [1, 3, 5, 4], [0.0, 0.5, 1.0, 0.75]),
    ("likert 1-10", (1.0, 10.0), [1, 10, 5.5], [0.0, 1.0, 0.5]),
    ("continuous [0,1]", (0.0, 1.0), [0.0, 0.25, 0.9, 1.0], [0.0, 0.25, 0.9, 1.0]),
    ("percentage 0-100", (0.0, 100.0), [0, 50, 100], [0.0, 0.5, 1.0]),
    ("negative-anchored -1..1", (-1.0, 1.0), [-1, 0, 1], [0.0, 0.5, 1.0]),
]


@pytest.mark.parametrize(
    ("name", "scale", "raw", "expected"), SCORE_TYPES, ids=[s[0] for s in SCORE_TYPES]
)
def test_score_types_normalise_to_unit_interval(
    tmp_path: Path, name: str, scale: tuple[float, float], raw: list[float], expected: list[float]
) -> None:
    path = write_jsonl(tmp_path, [{"item_id": f"q{i}", "score": s} for i, s in enumerate(raw)])
    obs = load(path, scale)
    assert [o.score for o in obs] == pytest.approx(expected)
    # The raw value and its scale are retained, so normalisation is reversible/auditable.
    assert [o.raw_score for o in obs] == pytest.approx(raw)
    assert all(o.scale == scale for o in obs)
    assert [o.denormalised() for o in obs] == pytest.approx(raw)


def test_booleans_are_accepted_as_a_binary_rubric(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [{"item_id": "a", "score": True}, {"item_id": "b", "score": False}])
    assert [o.score for o in load(path, (0.0, 1.0))] == [1.0, 0.0]


def test_item_and_score_aliases(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [{"id": "a", "value": 3}, {"test_id": "b", "score": 5}])
    obs = load(path, (1.0, 5.0))
    assert [(o.item_id, o.score) for o in obs] == [("a", 0.5), ("b", 1.0)]


def test_explicit_field_names_override_aliases(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [{"case": "a", "grade": 4}])
    obs = load(path, (1.0, 5.0), item_field="case", score_field="grade")
    assert obs[0].item_id == "a"
    assert obs[0].score == pytest.approx(0.75)


def test_blank_lines_and_comments_are_skipped(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path, ['{"item_id": "a", "score": 1}', "", "// a comment", '{"item_id": "b", "score": 5}']
    )
    assert len(load(path, (1.0, 5.0))) == 2


# --- malformed -------------------------------------------------------------------------
# (name, rows, scale, substring the error must contain, line it must be reported on)
MALFORMED: list[tuple[str, list[dict | str], tuple[float, float], str, int]] = [
    (
        "score above the declared scale",
        [{"item_id": "a", "score": 3}, {"item_id": "b", "score": 7}],
        (1.0, 5.0),
        "outside the declared score_scale [1.0, 5.0]",
        2,
    ),
    (
        "score below the declared scale",
        [{"item_id": "a", "score": 0}],
        (1.0, 5.0),
        "outside the declared score_scale",
        1,
    ),
    (
        "score is not a number",
        [{"item_id": "a", "score": "good"}],
        (1.0, 5.0),
        "is not a number",
        1,
    ),
    (
        "score is null",
        [{"item_id": "a", "score": None}],
        (1.0, 5.0),
        "is not a number",
        1,
    ),
    (
        "missing score field",
        [{"item_id": "a", "rating": 3}],
        (1.0, 5.0),
        "has no score field",
        1,
    ),
    (
        "missing item id",
        [{"score": 3}],
        (1.0, 5.0),
        "no item id field",
        1,
    ),
    (
        "duplicate item id",
        [{"item_id": "a", "score": 3}, {"item_id": "a", "score": 4}],
        (1.0, 5.0),
        "duplicate item id",
        2,
    ),
    (
        "not valid json",
        ['{"item_id": "a", "score": 3}', "{not json}"],
        (1.0, 5.0),
        "not valid JSON",
        2,
    ),
    (
        "line is not an object",
        ["[1, 2, 3]"],
        (1.0, 5.0),
        "expected a JSON object",
        1,
    ),
    (
        "NaN score",
        ['{"item_id": "a", "score": NaN}'],
        (1.0, 5.0),
        "not finite",
        1,
    ),
    (
        "empty file",
        [],
        (1.0, 5.0),
        "no observations found",
        0,
    ),
    (
        "ambiguous conflicting aliases",
        [{"item_id": "a", "id": "b", "score": 3}],
        (1.0, 5.0),
        "ambiguous fields",
        1,
    ),
]


@pytest.mark.parametrize(
    ("name", "rows", "scale", "needle", "line"), MALFORMED, ids=[m[0] for m in MALFORMED]
)
def test_malformed_rows_are_rejected_with_a_specific_message(
    tmp_path: Path,
    name: str,
    rows: list[dict | str],
    scale: tuple[float, float],
    needle: str,
    line: int,
) -> None:
    path = write_jsonl(tmp_path, rows) if rows else write_jsonl(tmp_path, [""])
    with pytest.raises(IngestError) as excinfo:
        load(path, scale)
    err = excinfo.value
    rendered = err.render()
    assert needle in rendered, f"{name}: expected {needle!r} in:\n{rendered}"
    assert str(path) in rendered
    assert any(i.line == line for i in err.issues), (
        f"{name}: expected an issue on line {line}, got {[i.line for i in err.issues]}"
    )
    assert all(i.hint for i in err.issues), f"{name}: an issue has no fix hint"


def test_at_least_four_malformed_cases() -> None:
    assert len(MALFORMED) >= 4


def test_missing_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(IngestError) as excinfo:
        load(tmp_path / "nope.jsonl", (0.0, 1.0))
    assert "file not found" in excinfo.value.render()


def test_every_problem_is_reported_not_just_the_first(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path,
        [
            {"item_id": "a", "score": 99},
            {"item_id": "b", "score": "x"},
            {"score": 3},
        ],
    )
    with pytest.raises(IngestError) as excinfo:
        load(path, (1.0, 5.0))
    assert len(excinfo.value.issues) == 3


def test_out_of_range_is_never_clamped(tmp_path: Path) -> None:
    # Clamping would compress a real signal into the bound and make a drifting
    # stream look stable. It must raise instead (Hard Rule 10).
    path = write_jsonl(tmp_path, [{"item_id": "a", "score": 12}])
    with pytest.raises(IngestError):
        load(path, (1.0, 5.0))


def test_observation_rejects_out_of_unit_interval_directly() -> None:
    with pytest.raises(ValueError, match="outside the declared score_scale"):
        Observation.normalised("a", 9.0, (1.0, 5.0))
    with pytest.raises(ValueError, match="outside"):
        Observation(item_id="a", score=1.5, raw_score=9.0, scale=(1.0, 5.0))


def test_suite_hash_is_order_independent_and_membership_sensitive() -> None:
    assert suite_hash_of(["b", "a", "c"]) == suite_hash_of(["a", "b", "c"])
    assert suite_hash_of(["a", "b"]) != suite_hash_of(["a", "b", "c"])
