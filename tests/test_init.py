"""Phase 0.6 verify: detection over four project shapes; an existing config is never lost."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchlock.adapters import detect_framework, infer_score_scale
from benchlock.cli import EXIT_ERROR, EXIT_OK, app
from benchlock.config import DEFAULT_CONFIG_NAME, AdapterKind, BenchlockConfig

runner = CliRunner()


def write_scores(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps({"item_id": f"q{i}", "score": v}) for i, v in enumerate(values)) + "\n"
    )


# --- the four project shapes ---------------------------------------------------------------


@pytest.fixture
def promptfoo_project(tmp_path: Path) -> Path:
    (tmp_path / "promptfooconfig.yaml").write_text("providers:\n  - anthropic:messages:claude\n")
    (tmp_path / "output.json").write_text("{}")
    return tmp_path


@pytest.fixture
def inspect_project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["inspect-ai>=0.3"]\n')
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "2026-09-01T00-00-00_task.eval").write_bytes(b"PK\x03\x04")
    return tmp_path


@pytest.fixture
def deepeval_project(tmp_path: Path) -> Path:
    (tmp_path / ".deepeval").write_text("{}")
    (tmp_path / "requirements.txt").write_text("deepeval==2.0.0\n")
    return tmp_path


@pytest.fixture
def bare_jsonl_project(tmp_path: Path) -> Path:
    write_scores(tmp_path / "evals" / "run1.jsonl", [1, 3, 5, 4, 2])
    write_scores(tmp_path / "evals" / "run2.jsonl", [2, 3, 5, 5, 1])
    return tmp_path


SHAPES = [
    ("promptfoo_project", AdapterKind.PROMPTFOO, "promptfooconfig.yaml"),
    ("inspect_project", AdapterKind.INSPECT_AI, ".eval log"),
    ("deepeval_project", AdapterKind.DEEPEVAL, ".deepeval"),
    ("bare_jsonl_project", AdapterKind.JSONL, "JSONL"),
]


@pytest.mark.parametrize(("fixture", "expected", "evidence"), SHAPES, ids=[s[0] for s in SHAPES])
def test_detects_each_project_shape(
    request: pytest.FixtureRequest, fixture: str, expected: AdapterKind, evidence: str
) -> None:
    root: Path = request.getfixturevalue(fixture)
    found = detect_framework(root)
    assert found.adapter is expected
    assert found.confident
    assert any(evidence in line for line in found.evidence), found.evidence


@pytest.mark.parametrize(("fixture", "expected", "evidence"), SHAPES, ids=[s[0] for s in SHAPES])
def test_init_round_trips_for_each_shape(
    request: pytest.FixtureRequest, fixture: str, expected: AdapterKind, evidence: str
) -> None:
    root: Path = request.getfixturevalue(fixture)
    result = runner.invoke(app, ["init", str(root)])
    assert result.exit_code == EXIT_OK, result.output

    target = root / DEFAULT_CONFIG_NAME
    assert target.exists()
    # The generated config parses, and round-trips through dump/load unchanged.
    cfg = BenchlockConfig.load(target)
    assert cfg.system.adapter is expected
    assert BenchlockConfig.parse(cfg.to_yaml()) == cfg
    # It stays readable: the evidence is written into the file as comments.
    assert "# Written by `benchlock init`" in target.read_text()


def test_empty_project_falls_back_to_jsonl_and_says_so(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == EXIT_OK
    assert "no eval framework" in result.output
    cfg = BenchlockConfig.load(tmp_path / DEFAULT_CONFIG_NAME)
    assert cfg.system.adapter is AdapterKind.JSONL


# --- never lose an existing config ----------------------------------------------------------


def test_existing_config_is_backed_up_byte_for_byte(bare_jsonl_project: Path) -> None:
    target = bare_jsonl_project / DEFAULT_CONFIG_NAME
    original = b"version: 1\nalpha: 0.01\n# my careful hand-written notes\n"
    target.write_bytes(original)

    result = runner.invoke(app, ["init", str(bare_jsonl_project)])
    # Without --force it refuses, and the original is untouched.
    assert result.exit_code == EXIT_ERROR
    assert target.read_bytes() == original
    backups = list(bare_jsonl_project.glob("benchlock.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_force_replaces_but_still_backs_up(bare_jsonl_project: Path) -> None:
    target = bare_jsonl_project / DEFAULT_CONFIG_NAME
    original = b"version: 1\nalpha: 0.01\n# hand-written\n"
    target.write_bytes(original)

    result = runner.invoke(app, ["init", str(bare_jsonl_project), "--force"])
    assert result.exit_code == EXIT_OK
    assert target.read_bytes() != original
    backups = list(bare_jsonl_project.glob("benchlock.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original, "the backup must be byte-for-byte"
    assert BenchlockConfig.load(target).system.adapter is AdapterKind.JSONL


def test_init_on_a_file_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "notadir.txt"
    f.write_text("x")
    result = runner.invoke(app, ["init", str(f)])
    assert result.exit_code == EXIT_ERROR
    assert "not a directory" in result.output


# --- score scale inference -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 1, 1, 0], (0.0, 1.0)),
        ([1, 3, 5], (1.0, 5.0)),
        ([1, 7, 10], (1.0, 10.0)),
        ([0, 55, 100], (0.0, 100.0)),
        ([0.0, 0.5, 1.0], (0.0, 1.0)),
    ],
)
def test_score_scale_is_inferred_as_the_tightest_standard_range(
    tmp_path: Path, values: list[float], expected: tuple[float, float]
) -> None:
    write_scores(tmp_path / "run.jsonl", values)
    scale, note = infer_score_scale(tmp_path)
    assert scale == expected
    assert "CONFIRM THIS" in note, "an inferred scale must be presented as a guess"


def test_nonstandard_range_is_reported_rather_than_guessed(tmp_path: Path) -> None:
    write_scores(tmp_path / "run.jsonl", [-40, 0, 733])
    scale, note = infer_score_scale(tmp_path)
    assert scale is None
    assert "matches no standard rubric range" in note


def test_inferred_scale_reaches_the_written_config(bare_jsonl_project: Path) -> None:
    result = runner.invoke(app, ["init", str(bare_jsonl_project)])
    assert result.exit_code == EXIT_OK
    assert BenchlockConfig.load(bare_jsonl_project / DEFAULT_CONFIG_NAME).score_scale == (1.0, 5.0)
    assert "CONFIRM THIS" in result.output


def test_the_ledger_is_not_mistaken_for_eval_output(tmp_path: Path) -> None:
    (tmp_path / ".benchlock").mkdir()
    (tmp_path / ".benchlock" / "ledger.jsonl").write_text('{"item_id":"a","score":1}\n')
    found = detect_framework(tmp_path)
    assert not found.confident
