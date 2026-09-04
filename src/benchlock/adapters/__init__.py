"""Ingest adapters, and the detection that picks one for you.

`benchlock init` uses `detect_framework` to look at a project and guess what it already
runs, so the generated config is pre-filled rather than a blank form. Detection is
best-effort by nature: every guess it makes is reported as evidence in the written
config, so the user can see *why* it chose what it chose and correct it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from benchlock.config import AdapterKind

#: Common rubric ranges, smallest first. `infer_score_scale` picks the tightest standard
#: range that contains every observed value, and always reports it as a guess.
STANDARD_SCALES: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),
    (1.0, 5.0),
    (0.0, 5.0),
    (1.0, 7.0),
    (1.0, 10.0),
    (0.0, 10.0),
    (0.0, 100.0),
)

_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".benchlock", "dist", "build"}


@dataclass(frozen=True, slots=True)
class Detection:
    """What we found, and why we think so."""

    adapter: AdapterKind
    path: Path
    evidence: tuple[str, ...] = ()
    judge_model: str | None = None
    score_scale: tuple[float, float] | None = None
    score_scale_note: str = ""
    confident: bool = False


@dataclass
class _Signals:
    hits: list[str] = field(default_factory=list)
    path: Path | None = None


def _iter_files(root: Path, patterns: tuple[str, ...], limit: int = 4000) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        for p in root.rglob(pattern):
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            found.append(p)
            if len(found) >= limit:
                return found
    return found


def _read_text(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _detect_promptfoo(root: Path) -> _Signals:
    sig = _Signals()
    for name in ("promptfooconfig.yaml", "promptfooconfig.yml", "promptfoo.yaml", "promptfoo.yml"):
        candidate = root / name
        if candidate.exists():
            sig.hits.append(f"found {name}")
            sig.path = root / "output.json" if (root / "output.json").exists() else root
    pkg = root / "package.json"
    if pkg.exists() and "promptfoo" in _read_text(pkg):
        sig.hits.append("package.json depends on promptfoo")
        sig.path = sig.path or root
    if (root / ".promptfoo").is_dir():
        sig.hits.append("found .promptfoo/")
        sig.path = sig.path or root / ".promptfoo"
    return sig


def _detect_inspect_ai(root: Path) -> _Signals:
    sig = _Signals()
    eval_logs = _iter_files(root, ("*.eval",), limit=5)
    if eval_logs:
        sig.hits.append(f"found {len(eval_logs)} Inspect AI .eval log(s)")
        sig.path = eval_logs[0].parent
    for meta in ("pyproject.toml", "requirements.txt"):
        path = root / meta
        if path.exists() and "inspect_ai" in _read_text(path).replace("-", "_"):
            sig.hits.append(f"{meta} depends on inspect_ai")
            sig.path = sig.path or root / "logs"
    if (root / "logs").is_dir() and _iter_files(root / "logs", ("*.json",), limit=1):
        sig.hits.append("found logs/ containing json")
        sig.path = sig.path or root / "logs"
    return sig


def _detect_deepeval(root: Path) -> _Signals:
    sig = _Signals()
    for name in (".deepeval", ".deepeval-cache.json", "deepeval.json"):
        if (root / name).exists():
            sig.hits.append(f"found {name}")
            sig.path = sig.path or root
    for meta in ("pyproject.toml", "requirements.txt"):
        path = root / meta
        if path.exists() and "deepeval" in _read_text(path):
            sig.hits.append(f"{meta} depends on deepeval")
            sig.path = sig.path or root
    return sig


def _detect_jsonl(root: Path) -> _Signals:
    """A bare JSONL of scores — the universal fallback."""
    sig = _Signals()
    candidates = [
        p
        for p in _iter_files(root, ("*.jsonl",), limit=50)
        if p.name != "ledger.jsonl" and "fixtures" not in p.parts
    ]
    scored = [p for p in candidates if _looks_like_scores(p)]
    if scored:
        best = sorted(scored, key=lambda p: (len(p.parts), p.name))[0]
        sig.hits.append(f"found {len(scored)} JSONL file(s) with item/score fields")
        sig.path = best.parent if len(scored) > 1 else best
    return sig


def _looks_like_scores(path: Path) -> bool:
    from benchlock.adapters.jsonl import ITEM_ALIASES, SCORE_ALIASES

    try:
        with path.open(encoding="utf-8") as fh:
            for _ in range(5):
                line = fh.readline()
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and any(a in obj for a in SCORE_ALIASES):
                    return any(a in obj for a in ITEM_ALIASES)
    except (OSError, json.JSONDecodeError):
        return False
    return False


def infer_score_scale(root: Path) -> tuple[tuple[float, float] | None, str]:
    """Suggest a score_scale from observed values. Always a suggestion, never a decision.

    Returns the tightest standard range containing every value seen, plus a note saying
    how many values it looked at. Hard Rule 10 keeps the scale *declared*: this only
    pre-fills the declaration so the user has something concrete to confirm or correct.
    """
    from benchlock.adapters.jsonl import SCORE_ALIASES

    values: list[float] = []
    for path in _iter_files(root, ("*.jsonl",), limit=20):
        if path.name == "ledger.jsonl":
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    for alias in SCORE_ALIASES:
                        raw = obj.get(alias)
                        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                            values.append(float(raw))
                            break
                    if len(values) >= 5000:
                        break
        except (OSError, json.JSONDecodeError):
            continue
    if not values:
        return None, ""
    lo, hi = min(values), max(values)
    for scale in STANDARD_SCALES:
        if scale[0] <= lo and hi <= scale[1]:
            return scale, (
                f"inferred from {len(values)} score(s) spanning [{lo:g}, {hi:g}] — CONFIRM THIS"
            )
    return None, f"observed scores span [{lo:g}, {hi:g}], which matches no standard rubric range"


def detect_framework(root: Path) -> Detection:
    """Pick the adapter this project most likely needs."""
    probes: list[tuple[AdapterKind, _Signals]] = [
        (AdapterKind.PROMPTFOO, _detect_promptfoo(root)),
        (AdapterKind.INSPECT_AI, _detect_inspect_ai(root)),
        (AdapterKind.DEEPEVAL, _detect_deepeval(root)),
        (AdapterKind.JSONL, _detect_jsonl(root)),
    ]
    scale, note = infer_score_scale(root)
    for kind, sig in probes:
        if sig.hits:
            return Detection(
                adapter=kind,
                path=sig.path or root,
                evidence=tuple(sig.hits),
                score_scale=scale,
                score_scale_note=note,
                confident=True,
            )
    return Detection(
        adapter=AdapterKind.JSONL,
        path=root / "evals" / "results",
        evidence=("no eval framework detected; defaulting to the universal JSONL adapter",),
        score_scale=scale,
        score_scale_note=note,
        confident=False,
    )
