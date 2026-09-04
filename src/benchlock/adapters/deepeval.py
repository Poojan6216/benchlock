"""Ingest DeepEval results.

DeepEval writes test-run JSON containing a list of test cases, each with `metricsData`
(or `metrics_data` / `metrics`) holding one entry per metric with a `score`. A run with
several metrics is ambiguous, so it is refused rather than averaged: averaging metrics is a
modelling decision that belongs to the person who chose them, not to an ingest adapter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchlock.adapters.jsonl import IngestError, IngestIssue, parse_records
from benchlock.model.streams import Observation

CASE_KEYS = ("testCases", "test_cases", "testResults", "test_results", "results")
METRIC_KEYS = ("metricsData", "metrics_data", "metrics", "metricsMetadata")
NAME_KEYS = ("name", "identifier", "test_name", "input")


def _refuse(source: str, message: str, hint: str) -> IngestError:
    return IngestError(source, [IngestIssue(line=0, message=message, hint=hint)])


def _cases(data: Any, source: str) -> list[Mapping[str, Any]]:
    if isinstance(data, list):
        return list(data)
    if not isinstance(data, dict):
        raise _refuse(
            source,
            f"expected a DeepEval JSON object or array, got {type(data).__name__}",
            "point `system.path` at DeepEval's test-run JSON",
        )
    for key in CASE_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            return list(value)
    keys = ", ".join(sorted(data)) or "(none)"
    raise _refuse(
        source,
        f"no test-case array found (looked for {', '.join(CASE_KEYS)}). Top-level keys: {keys}",
        "benchlock refuses to guess at an unrecognised schema. Export DeepEval's test-run "
        "JSON, or convert to the universal JSONL adapter and set `adapter: jsonl`",
    )


def _score_of(case: Mapping[str, Any], index: int, source: str, metric: str | None) -> float:
    for key in METRIC_KEYS:
        metrics = case.get(key)
        if not isinstance(metrics, list) or not metrics:
            continue
        named = {
            str(m.get("name", f"metric{i}")): m
            for i, m in enumerate(metrics)
            if isinstance(m, Mapping)
        }
        if metric is not None:
            if metric not in named:
                raise _refuse(
                    source,
                    f"case {index} has no metric named {metric!r} "
                    f"(present: {', '.join(sorted(named))})",
                    "check the metric name",
                )
            chosen = named[metric]
        elif len(named) == 1:
            chosen = next(iter(named.values()))
        else:
            raise _refuse(
                source,
                f"case {index} has {len(named)} metrics ({', '.join(sorted(named))}) and "
                "none was selected",
                "benchlock monitors one score stream at a time. Averaging your metrics is "
                "a modelling decision that belongs to you, not to an ingest adapter — pick "
                "one metric, or run one benchlock config per metric",
            )
        value = chosen.get("score")
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        raise _refuse(
            source,
            f"case {index}: metric score is {value!r}, not a number",
            "DeepEval metrics should report a numeric `score`",
        )
    if isinstance(case.get("score"), (int, float)):
        return float(case["score"])
    if isinstance(case.get("success"), bool):
        return float(case["success"])
    raise _refuse(
        source,
        f"case {index} has no metric scores; keys present: {', '.join(sorted(case))}",
        "check that the DeepEval run produced metrics",
    )


def load(
    path: Path, scale: tuple[float, float], *, metric: str | None = None
) -> tuple[Observation, ...]:
    if not path.exists():
        raise _refuse(
            str(path), "file not found", "point `system.path` at DeepEval's test-run JSON"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _refuse(
            str(path), f"not valid JSON: {exc.msg}", "expected DeepEval's JSON output"
        ) from exc

    cases = _cases(data, str(path))
    if not cases:
        raise _refuse(
            str(path), "the DeepEval output contains no test cases", "check the DeepEval run"
        )
    records: list[tuple[int, dict[str, Any]]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise _refuse(
                str(path), f"case {index} is not an object", "each test case should be an object"
            )
        item_id = next(
            (
                str(case[key]).strip()
                for key in NAME_KEYS
                if isinstance(case.get(key), str) and str(case[key]).strip()
            ),
            f"deepeval-case-{index}",
        )
        records.append(
            (index + 1, {"item_id": item_id, "score": _score_of(case, index, str(path), metric)})
        )
    return parse_records(records, scale, source=str(path))
