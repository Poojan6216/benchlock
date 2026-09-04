"""Ingest promptfoo eval output.

promptfoo has shipped several output shapes across major versions, and the per-result
schema is not fully published. So this adapter **identifies the shape explicitly and
refuses anything it does not recognise**, rather than reaching hopefully for a `score` key
somewhere in the tree. Guessing here would be worse than failing: a wrong field silently
produces a stream of plausible numbers that mean something else, and every verdict built on
top of it is confidently wrong.

Supported shapes:

* **v3+ nested** — ``{"version": 3, "results": {"results": [...]}}``
* **v2 flat** — ``{"version": 2, "results": [...]}``
* **bare list** — a JSON array of result objects

Within a result, the score is taken from ``score``, then ``gradingResult.score``, then
``namedScores`` when it holds exactly one metric. An item id comes from ``testCase.description``,
then a ``vars`` field, then the test index — with the index used only as a last resort,
because an id that is not stable across runs makes the anchor set meaningless.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchlock.adapters.jsonl import IngestError, IngestIssue, parse_records
from benchlock.model.streams import Observation

#: Fields checked, in order, for a stable per-test identifier.
ID_FIELDS = ("description", "id", "item_id", "test_id", "name")


def _refuse(source: str, message: str, hint: str) -> IngestError:
    return IngestError(source, [IngestIssue(line=0, message=message, hint=hint)])


def _results_array(data: Any, source: str) -> list[Mapping[str, Any]]:
    """Find the array of per-test results, or refuse with what was actually seen."""
    if isinstance(data, list):
        return list(data)
    if not isinstance(data, dict):
        raise _refuse(
            source,
            f"expected a promptfoo JSON object or array, got {type(data).__name__}",
            "point `system.path` at the file written by `promptfoo eval -o output.json`",
        )
    results = data.get("results")
    if isinstance(results, dict) and isinstance(results.get("results"), list):
        return list(results["results"])  # v3+
    if isinstance(results, list):
        return list(results)  # v2
    keys = ", ".join(sorted(data)) or "(none)"
    raise _refuse(
        source,
        f"this does not look like promptfoo output: no results array found. Top-level "
        f"keys are: {keys}",
        "benchlock refuses to guess at an unrecognised schema — a wrong field would "
        "produce plausible numbers that mean something else. Export with "
        "`promptfoo eval -o output.json`, or convert to the universal JSONL adapter "
        '(one {"item_id", "score"} object per line) and set `adapter: jsonl`',
    )


def _score_of(entry: Mapping[str, Any], index: int, source: str) -> float:
    if isinstance(entry.get("score"), (int, float)) and not isinstance(entry.get("score"), bool):
        return float(entry["score"])
    grading = entry.get("gradingResult")
    if isinstance(grading, Mapping) and isinstance(grading.get("score"), (int, float)):
        return float(grading["score"])
    named = entry.get("namedScores")
    if isinstance(named, Mapping) and len(named) == 1:
        (value,) = named.values()
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(named, Mapping) and len(named) > 1:
        raise _refuse(
            source,
            f"result {index} has {len(named)} named scores ({', '.join(sorted(named))}) and "
            "no overall `score`",
            "benchlock monitors one score stream at a time. Pick the metric you care about "
            "and export it as the top-level `score`, or run one benchlock config per metric",
        )
    # `success` alone is a legitimate binary rubric, but only if it is genuinely all there is.
    if isinstance(entry.get("success"), bool):
        return float(entry["success"])
    raise _refuse(
        source,
        f"result {index} has no numeric score (looked at `score`, `gradingResult.score`, "
        f"`namedScores`, `success`); keys present: {', '.join(sorted(entry))}",
        "check that your promptfoo assertions produce a score",
    )


def _item_id_of(entry: Mapping[str, Any], index: int) -> str:
    test_case = entry.get("testCase")
    if isinstance(test_case, Mapping):
        for field in ID_FIELDS:
            value = test_case.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        variables = test_case.get("vars")
        if isinstance(variables, Mapping):
            for field in ID_FIELDS:
                value = variables.get(field)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    for field in ID_FIELDS:
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Positional fallback. Stable only while the suite's order is stable, which is exactly
    # the assumption lattice rule 2 checks, so a reordering surfaces as suite drift rather
    # than as silent nonsense.
    return f"promptfoo-test-{index}"


def load(path: Path, scale: tuple[float, float]) -> tuple[Observation, ...]:
    """Read promptfoo output and normalise it through the universal JSONL contract."""
    if not path.exists():
        raise _refuse(
            str(path),
            "file not found",
            "run `promptfoo eval -o output.json` and point `system.path` at the result",
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _refuse(
            str(path), f"not valid JSON: {exc.msg}", "the file should be promptfoo's JSON export"
        ) from exc

    entries = _results_array(data, str(path))
    if not entries:
        raise _refuse(
            str(path),
            "the promptfoo output contains no results",
            "the eval produced no test cases; check the promptfoo run itself",
        )
    records: list[tuple[int, dict[str, Any]]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise _refuse(
                str(path),
                f"result {index} is a {type(entry).__name__}, not an object",
                "the results array should hold one object per test case",
            )
        records.append(
            (
                index + 1,
                {
                    "item_id": _item_id_of(entry, index),
                    "score": _score_of(entry, index, str(path)),
                },
            )
        )
    return parse_records(records, scale, source=str(path))


def detect_scale(path: Path) -> tuple[float, float] | None:
    """Best guess at the score range, for `benchlock init`. Always a suggestion."""
    try:
        entries = _results_array(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, json.JSONDecodeError, IngestError):
        return None
    values: list[float] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, Mapping):
            try:
                values.append(_score_of(entry, index, str(path)))
            except IngestError:
                return None
    if not values:
        return None
    return (0.0, 1.0) if max(values) <= 1.0 else (min(values), max(values))
