"""The universal ingest: a JSONL file of ``{item_id, score}``.

This is the lowest common denominator. Every framework adapter in Phase 8 normalises
into this contract, so there is exactly one place that decides what a valid score is.

Two rules do the work here, and both are Hard Rule 10 (fail loud):

* A score outside the declared ``score_scale`` is an **error**, never a clamp. Clamping
  would silently compress a real signal into the bound and make the stream look stable.
* Every failure names the file, the line, and what to do about it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchlock.model.streams import Observation

#: Accepted spellings for the two required fields. Aliases are a convenience, but an
#: ambiguous line (two different aliases present with different values) is an error.
ITEM_ALIASES = ("item_id", "id", "test_id", "case_id")
SCORE_ALIASES = ("score", "value")


@dataclass(frozen=True, slots=True)
class IngestIssue:
    line: int
    message: str
    hint: str


class IngestError(Exception):
    """Raised when a score file cannot be ingested faithfully."""

    def __init__(self, source: str, issues: Sequence[IngestIssue]) -> None:
        self.source = source
        self.issues = tuple(issues)
        super().__init__(self.render())

    def render(self) -> str:
        head = f"{len(self.issues)} problem(s) ingesting {self.source}:"
        body = [
            f"  {self.source}:{i.line}: {i.message}\n    fix: {i.hint}"
            if i.line
            else f"  {self.source}: {i.message}\n    fix: {i.hint}"
            for i in self.issues
        ]
        return "\n".join([head, *body])


class _AmbiguousFieldError(Exception):
    """Internal: two aliases present with different values. Carries a reportable issue."""

    def __init__(self, issue: IngestIssue) -> None:
        self.issue = issue
        super().__init__(issue.message)


def _pick(record: Mapping[str, Any], aliases: Sequence[str], line: int) -> tuple[str, Any] | None:
    """Return the single (key, value) among ``aliases`` present, or None.

    Two aliases carrying *different* values is ambiguous and raises: guessing which the
    user meant is exactly the failure mode Phase 8.1 forbids.
    """
    found = [(a, record[a]) for a in aliases if a in record]
    if not found:
        return None
    distinct = {json.dumps(v, sort_keys=True, default=str) for _, v in found}
    if len(distinct) > 1:
        names = ", ".join(f"{k}={v!r}" for k, v in found)
        raise _AmbiguousFieldError(
            IngestIssue(
                line=line,
                message=f"ambiguous fields with different values: {names}",
                hint=f"keep exactly one of {', '.join(aliases)} per record",
            )
        )
    return found[0]


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (1-based line number, object) for each non-blank line."""
    issues: list[IngestIssue] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            text = raw.strip()
            if not text or text.startswith("//"):
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                issues.append(
                    IngestIssue(
                        line=line_no,
                        message=f"not valid JSON: {exc.msg}",
                        hint="each line must be one complete JSON object",
                    )
                )
                continue
            if not isinstance(obj, dict):
                issues.append(
                    IngestIssue(
                        line=line_no,
                        message=f"expected a JSON object, got {type(obj).__name__}",
                        hint='each line should look like {"item_id": "q1", "score": 4}',
                    )
                )
                continue
            yield line_no, obj
    if issues:
        raise IngestError(str(path), issues)


def parse_records(
    records: Iterable[tuple[int, Mapping[str, Any]]],
    scale: tuple[float, float],
    *,
    source: str,
    item_field: str | None = None,
    score_field: str | None = None,
) -> tuple[Observation, ...]:
    """Normalise raw records into Observations, collecting every problem before raising."""
    lo, hi = scale
    issues: list[IngestIssue] = []
    seen: dict[str, int] = {}
    out: list[Observation] = []

    for line_no, record in records:
        # --- item id ---
        if item_field is not None:
            item = (item_field, record[item_field]) if item_field in record else None
            aliases: Sequence[str] = (item_field,)
        else:
            aliases = ITEM_ALIASES
            try:
                item = _pick(record, ITEM_ALIASES, line_no)
            except _AmbiguousFieldError as exc:
                issues.append(exc.issue)
                continue
        if item is None:
            issues.append(
                IngestIssue(
                    line=line_no,
                    message=(
                        f"no item id field; looked for {', '.join(aliases)}. "
                        f"keys present: {', '.join(sorted(record)) or '(none)'}"
                    ),
                    hint="add an `item_id` to every record; it must be stable across runs",
                )
            )
            continue
        item_id = str(item[1])
        if not item_id or item[1] is None:
            issues.append(
                IngestIssue(
                    line=line_no,
                    message="item id is empty",
                    hint="`item_id` must be a non-empty, stable identifier",
                )
            )
            continue
        if item_id in seen:
            issues.append(
                IngestIssue(
                    line=line_no,
                    message=f"duplicate item id {item_id!r} (first seen on line {seen[item_id]})",
                    hint=(
                        "each item may be scored once per run; if you score replicates, "
                        "ingest them as separate runs or give them distinct ids"
                    ),
                )
            )
            continue

        # --- score ---
        if score_field is not None:
            found = (score_field, record[score_field]) if score_field in record else None
            score_aliases: Sequence[str] = (score_field,)
        else:
            score_aliases = SCORE_ALIASES
            try:
                found = _pick(record, SCORE_ALIASES, line_no)
            except _AmbiguousFieldError as exc:
                issues.append(exc.issue)
                continue
        if found is None:
            issues.append(
                IngestIssue(
                    line=line_no,
                    message=(
                        f"item {item_id!r} has no score field; looked for "
                        f"{', '.join(score_aliases)}. keys present: {', '.join(sorted(record))}"
                    ),
                    hint="every record needs a numeric `score`",
                )
            )
            continue
        raw = found[1]
        if isinstance(raw, bool):
            raw_score = float(raw)  # a pass/fail suite is a legitimate binary rubric
        elif isinstance(raw, (int, float)):
            raw_score = float(raw)
        else:
            issues.append(
                IngestIssue(
                    line=line_no,
                    message=f"item {item_id!r}: score {raw!r} is not a number",
                    hint=(
                        "scores must be numeric; map categorical judgments to numbers in "
                        "your eval, and declare the range in `score_scale`"
                    ),
                )
            )
            continue
        if not math.isfinite(raw_score):
            issues.append(
                IngestIssue(
                    line=line_no,
                    message=f"item {item_id!r}: score is {raw_score}, which is not finite",
                    hint="NaN and infinity cannot be scored; drop or fix the item upstream",
                )
            )
            continue
        if not lo <= raw_score <= hi:
            issues.append(
                IngestIssue(
                    line=line_no,
                    message=(
                        f"item {item_id!r}: score {raw_score} is outside the declared "
                        f"score_scale [{lo}, {hi}]"
                    ),
                    hint=(
                        f"either fix the rubric so it stays inside [{lo}, {hi}], or change "
                        "`score_scale` in benchlock.yaml to the range it really emits. "
                        "benchlock will not clamp: clamping hides real movement"
                    ),
                )
            )
            continue

        seen[item_id] = line_no
        out.append(Observation.normalised(item_id, raw_score, scale))

    if issues:
        raise IngestError(source, issues)
    if not out:
        raise IngestError(
            source,
            [
                IngestIssue(
                    line=0,
                    message="no observations found",
                    hint=(
                        "the file should hold one JSON object per line, e.g. "
                        '{"item_id": "q1", "score": 4}'
                    ),
                )
            ],
        )
    return tuple(out)


def load(
    path: Path,
    scale: tuple[float, float],
    *,
    item_field: str | None = None,
    score_field: str | None = None,
) -> tuple[Observation, ...]:
    """Read a JSONL score file and return normalised observations."""
    if not path.exists():
        raise IngestError(
            str(path),
            [
                IngestIssue(
                    line=0,
                    message="file not found",
                    hint="check the path, or point `system.path` in benchlock.yaml at your results",
                )
            ],
        )
    return parse_records(
        iter_jsonl(path),
        scale,
        source=str(path),
        item_field=item_field,
        score_field=score_field,
    )
