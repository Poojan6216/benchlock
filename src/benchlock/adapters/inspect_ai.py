"""Ingest Inspect AI eval logs.

Inspect writes `.eval` files (a zip containing JSON) and, with `--log-format json`, plain
JSON logs. Both carry a `samples` array where each sample has an `id` and a `scores`
mapping of scorer name to a result object.

Inspect scorers commonly emit **letter grades** — `"C"` for correct, `"I"` for incorrect,
`"P"` partial — as well as numbers and booleans. Those are mapped explicitly below. An
unrecognised value is refused rather than coerced, because silently mapping an unknown
grade to a number is how a stream of meaningless scores gets built.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchlock.adapters.jsonl import IngestError, IngestIssue, parse_records
from benchlock.model.streams import Observation

#: Inspect's letter grades, and what each means numerically.
LETTER_GRADES: dict[str, float] = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}


def _refuse(source: str, message: str, hint: str) -> IngestError:
    return IngestError(source, [IngestIssue(line=0, message=message, hint=hint)])


def _read_log(path: Path) -> dict[str, Any]:
    """Read either a `.eval` archive or a plain JSON log."""
    if path.suffix == ".eval" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            # Inspect stores the log as a JSON member; find it rather than assume a name.
            candidates = [n for n in names if n.endswith(".json")]
            if not candidates:
                raise _refuse(
                    str(path),
                    f"the .eval archive contains no JSON log (members: {', '.join(names[:6])})",
                    "this does not look like an Inspect AI log; re-export with "
                    "`inspect eval --log-format json`",
                )
            merged: dict[str, Any] = {}
            for name in sorted(candidates):
                part = json.loads(archive.read(name))
                if isinstance(part, dict):
                    merged.update(part)
            return merged
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _refuse(
            str(path), f"not valid JSON: {exc.msg}", "expected an Inspect AI JSON log"
        ) from exc
    if not isinstance(data, dict):
        raise _refuse(
            str(path),
            f"expected an Inspect AI log object, got {type(data).__name__}",
            "point `system.path` at a `.eval` file or a JSON log",
        )
    return data


def _numeric(value: Any, sample_id: str, scorer: str, source: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        grade = LETTER_GRADES.get(value.strip().upper())
        if grade is not None:
            return grade
        raise _refuse(
            source,
            f"sample {sample_id!r}, scorer {scorer!r}: unrecognised grade {value!r}",
            f"benchlock maps Inspect's letter grades ({', '.join(sorted(LETTER_GRADES))}) "
            "and numbers. It will not invent a value for anything else — map the grade to "
            "a number in your scorer",
        )
    raise _refuse(
        source,
        f"sample {sample_id!r}, scorer {scorer!r}: score is {type(value).__name__}, not a "
        "number or a grade",
        "the scorer must produce a numeric or letter-graded value",
    )


def load(
    path: Path, scale: tuple[float, float], *, scorer: str | None = None
) -> tuple[Observation, ...]:
    """Read an Inspect AI log and normalise it through the universal JSONL contract."""
    if not path.exists():
        raise _refuse(
            str(path),
            "file not found",
            "point `system.path` at an Inspect AI `.eval` file or JSON log",
        )
    data = _read_log(path)
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        keys = ", ".join(sorted(data)) or "(none)"
        raise _refuse(
            str(path),
            f"no `samples` array in the log. Top-level keys are: {keys}",
            "benchlock refuses to guess at an unrecognised schema. Re-export with "
            "`inspect eval --log-format json`, or convert to the universal JSONL adapter",
        )

    records: list[tuple[int, dict[str, Any]]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise _refuse(
                str(path), f"sample {index} is not an object", "each sample should be a JSON object"
            )
        scores = sample.get("scores")
        if not isinstance(scores, Mapping) or not scores:
            raise _refuse(
                str(path),
                f"sample {index} has no `scores`",
                "the eval produced a sample with no score; check the Inspect run",
            )
        chosen = scorer
        if chosen is None:
            if len(scores) > 1:
                raise _refuse(
                    str(path),
                    f"sample {index} has {len(scores)} scorers "
                    f"({', '.join(sorted(scores))}) and none was selected",
                    "benchlock monitors one score stream at a time. Choose a scorer, or "
                    "run one benchlock config per scorer",
                )
            chosen = next(iter(scores))
        if chosen not in scores:
            raise _refuse(
                str(path),
                f"sample {index} has no scorer named {chosen!r} "
                f"(present: {', '.join(sorted(scores))})",
                "check the scorer name",
            )
        entry = scores[chosen]
        value = entry.get("value") if isinstance(entry, Mapping) else entry
        sample_id = str(sample.get("id", index))
        records.append(
            (
                index + 1,
                {
                    "item_id": sample_id,
                    "score": _numeric(value, sample_id, chosen, str(path)),
                },
            )
        )
    return parse_records(records, scale, source=str(path))
