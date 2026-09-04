"""Phase 0.2 verify: malformed configs must each name the field, the line, and the fix."""

from __future__ import annotations

import textwrap

import pytest

from benchlock.config import (
    SCHEMA_VERSION,
    AnchorMode,
    BenchlockConfig,
    ConfigError,
    load_yaml_with_lines,
)

VALID = textwrap.dedent("""\
    version: 1
    alpha: 0.05
    score_scale: [1, 5]
    min_runs: 8
    min_obs: 30

    system:
      adapter: promptfoo
      path: ./evals/results/

    anchor:
      mode: frozen-self
      n: 260
      cadence: 1
      selection: stratified
      noise_replicates: 5

    judge:
      provider: anthropic
      model: claude-sonnet-4-5-20250929
      rubric: ./evals/rubric.md
      params: { temperature: 0.0, max_tokens: 512 }

    gate:
      fail_on: [system, both]
      warn_on: [indeterminate]
    """)


def _mutate(line_starts_with: str, replacement: str) -> str:
    out = []
    for line in VALID.splitlines():
        out.append(replacement if line.strip().startswith(line_starts_with) else line)
    return "\n".join(out) + "\n"


def test_the_valid_config_parses() -> None:
    cfg = BenchlockConfig.parse(VALID)
    assert cfg.alpha == 0.05
    assert cfg.score_scale == (1.0, 5.0)
    assert cfg.anchor.n == 260
    assert cfg.judge.params.max_tokens == 512


def test_round_trip_load_dump_load_is_stable() -> None:
    once = BenchlockConfig.parse(VALID)
    twice = BenchlockConfig.parse(once.to_yaml())
    thrice = BenchlockConfig.parse(twice.to_yaml())
    assert once == twice == thrice
    assert twice.to_yaml() == thrice.to_yaml()


def test_defaults_round_trip() -> None:
    cfg = BenchlockConfig()
    assert BenchlockConfig.parse(cfg.to_yaml()) == cfg


# (yaml, expected field in the error, a substring the message or hint must contain)
MALFORMED: list[tuple[str, str, str, str]] = [
    (
        "unknown top-level key",
        VALID + "alpah: 0.1\n",
        "alpah",
        "unknown field",
    ),
    (
        "unknown nested key",
        _mutate("cadence:", "  cadance: 1"),
        "anchor.cadance",
        "unknown field",
    ),
    (
        "alpha above 0.5",
        _mutate("alpha:", "alpha: 0.9"),
        "alpha",
        "less than or equal to 0.5",
    ),
    (
        "alpha not a number",
        _mutate("alpha:", "alpha: loose"),
        "alpha",
        "valid number",
    ),
    (
        "alpha zero",
        _mutate("alpha:", "alpha: 0.0"),
        "alpha",
        "greater than 0",
    ),
    (
        "wrong schema version",
        _mutate("version:", "version: 7"),
        "version",
        f"version: {SCHEMA_VERSION}",
    ),
    (
        "score_scale reversed",
        _mutate("score_scale:", "score_scale: [5, 1]"),
        "<file>",
        "high > low",
    ),
    (
        "score_scale wrong arity",
        _mutate("score_scale:", "score_scale: [1, 2, 3]"),
        "score_scale",
        "2 items",
    ),
    (
        "min_runs below floor",
        _mutate("min_runs:", "min_runs: 1"),
        "min_runs",
        "greater than or equal to 2",
    ),
    (
        "unknown adapter",
        _mutate("adapter:", "  adapter: langsmith"),
        "system.adapter",
        "promptfoo",
    ),
    (
        "unknown anchor mode",
        _mutate("mode:", "  mode: vibes"),
        "anchor.mode",
        "frozen-self",
    ),
    (
        "anchor n is zero",
        _mutate("n:", "  n: 0"),
        "anchor.n",
        "benchlock plan",
    ),
    (
        "noise_replicates below 2",
        _mutate("noise_replicates:", "  noise_replicates: 1"),
        "anchor.noise_replicates",
        ">= 2",
    ),
    (
        "negative temperature",
        _mutate("params:", "  params: { temperature: -1.0, max_tokens: 512 }"),
        "judge.params.temperature",
        "greater than or equal to 0",
    ),
    (
        "unknown verdict in gate",
        _mutate("fail_on:", "  fail_on: [system, regressed]"),
        "gate.fail_on.1",
        "indeterminate",
    ),
    (
        "gate lists overlap",
        _mutate("warn_on:", "  warn_on: [system]"),
        "gate",
        "both fail_on and warn_on",
    ),
    (
        "stable would fail every build",
        _mutate("fail_on:", "  fail_on: [stable]"),
        "gate",
        "fail every healthy build",
    ),
    (
        "human mode without labels",
        _mutate("mode:", "  mode: human"),
        "anchor",
        "requires `labels:`",
    ),
    (
        "not valid yaml",
        "version: 1\nalpha: [1, 2\n",
        "<file>",
        "not valid YAML",
    ),
    (
        "empty file",
        "\n",
        "<file>",
        "empty",
    ),
    (
        "top level is a list",
        "- version: 1\n",
        "<file>",
        "must be a mapping",
    ),
]


@pytest.mark.parametrize(
    ("name", "text", "field", "needle"),
    MALFORMED,
    ids=[m[0] for m in MALFORMED],
)
def test_malformed_config_is_specific_and_actionable(
    name: str, text: str, field: str, needle: str
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        BenchlockConfig.parse(text, source="benchlock.yaml")
    err = excinfo.value
    rendered = err.render()

    fields = {issue.field for issue in err.issues}
    assert field in fields, f"{name}: expected an issue on `{field}`, got {sorted(fields)}"
    assert needle in rendered, f"{name}: expected {needle!r} in:\n{rendered}"
    # Every issue names the file and offers a fix.
    assert "benchlock.yaml" in rendered
    for issue in err.issues:
        assert issue.hint, f"{name}: issue on {issue.field} has no fix hint"


def test_there_are_at_least_twelve_malformed_cases() -> None:
    # The spec asks for 12+; keep the suite from being quietly trimmed.
    assert len(MALFORMED) >= 12


@pytest.mark.parametrize(
    ("text", "field", "expected_line"),
    [
        (_mutate("alpha:", "alpha: 0.9"), "alpha", 2),
        (_mutate("min_obs:", "min_obs: 0"), "min_obs", 5),
        (_mutate("n:", "  n: 0"), "anchor.n", 13),
        (_mutate("adapter:", "  adapter: langsmith"), "system.adapter", 8),
        (
            _mutate("params:", "  params: { temperature: 9.0, max_tokens: 512 }"),
            "judge.params.temperature",
            22,
        ),
        (_mutate("fail_on:", "  fail_on: [system, regressed]"), "gate.fail_on.1", 25),
    ],
)
def test_error_points_at_the_right_line(text: str, field: str, expected_line: int) -> None:
    with pytest.raises(ConfigError) as excinfo:
        BenchlockConfig.parse(text)
    issue = next(i for i in excinfo.value.issues if i.field == field)
    assert issue.line == expected_line, (
        f"expected line {expected_line} for `{field}`, got {issue.line}\n"
        + "\n".join(f"{n + 1:>3}: {ln}" for n, ln in enumerate(text.splitlines()))
    )


def test_line_map_covers_nested_and_sequence_paths() -> None:
    _, lines = load_yaml_with_lines(VALID, "benchlock.yaml")
    assert lines[("alpha",)] == 2
    assert lines[("score_scale", 0)] == 3
    assert lines[("anchor", "mode")] == 12
    assert lines[("gate", "fail_on", 1)] == 25


def test_missing_file_explains_how_to_create_one(tmp_path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        BenchlockConfig.load(tmp_path / "benchlock.yaml")
    assert "benchlock init" in excinfo.value.render()


def test_human_mode_with_labels_is_accepted() -> None:
    cfg = BenchlockConfig.parse(
        _mutate("mode:", "  mode: human\n  labels: ./evals/gold.jsonl"),
    )
    assert cfg.anchor.mode is AnchorMode.HUMAN
    assert cfg.anchor.labels is not None
