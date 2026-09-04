"""The Anthropic judge adapter.

The only place in the project that talks to a model, alongside its OpenAI sibling. It sends
a rubric and an answer, and reads back a number. Nothing here interprets, retries into a
different answer, or asks the model to reconsider — the judge produces data, and `decide()`
produces verdicts (Hard Rule 1).

The rubric must instruct the model to reply with a bare number. Anything else is a parse
failure and is raised, not guessed at: a judge whose output cannot be read is a broken
measurement, and silently coercing it would put an invented number into the ledger.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from benchlock.judge.base import (
    CostEstimate,
    JudgeDescription,
    JudgeRequest,
    JudgeResult,
    estimate_cost_for,
)
from benchlock.model.pins import JudgePin

#: Published prices per million tokens, used only for `benchlock plan`'s estimate. Out of
#: date prices make the estimate wrong, never the verdict.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-1": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class JudgeCallError(Exception):
    def __init__(self, message: str, hint: str = "") -> None:
        self.message = message
        self.hint = hint
        super().__init__(f"{message}" + (f"\n  fix: {hint}" if hint else ""))


def _price_for(model: str) -> tuple[float, float]:
    for prefix, price in PRICES.items():
        if model.startswith(prefix):
            return price
    return (3.0, 15.0)


def parse_score(text: str, scale: tuple[float, float]) -> float:
    """Read a bare number out of the judge's reply, or raise.

    Takes the *first* number in the response: a rubric that asks for a bare score and gets
    "4" parses cleanly, and one that gets "4 out of 5" still reads 4. A reply with no
    number at all is a failed measurement and says so.
    """
    match = _NUMBER.search(text.strip())
    if match is None:
        raise JudgeCallError(
            f"judge returned no number: {text[:120]!r}",
            "the rubric must instruct the judge to reply with a bare score and nothing else",
        )
    value = float(match.group())
    lo, hi = scale
    if not lo <= value <= hi:
        raise JudgeCallError(
            f"judge returned {value}, outside the declared score_scale [{lo}, {hi}]",
            "either fix the rubric so it stays inside the declared range, or change "
            "`score_scale` in benchlock.yaml. Benchlock will not clamp",
        )
    return value


@dataclass
class AnthropicJudge:
    """Provider adapter. Four methods, per `JudgeAdapter`."""

    model: str = "claude-sonnet-4-5-20250929"
    rubric_text: str = ""
    scale: tuple[float, float] = (1.0, 5.0)
    params: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0, "max_tokens": 16})
    api_key: str | None = None
    #: Set by Phase 7.9 to defeat provider-side response caching.
    nonce_template: str = "\n\n<!-- request-id: {nonce} -->"
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        self._client: Any = None

    # ---- the four methods -------------------------------------------------------------

    def describe(self) -> JudgeDescription:
        input_price, output_price = _price_for(self.model)
        return JudgeDescription(
            provider="anthropic",
            model=self.model,
            scale=self.scale,
            # Anthropic publishes dated snapshots, so `replicate` mode is available.
            supports_pinned_snapshots=bool(re.search(r"-\d{8}$", self.model)),
            input_cost_per_mtok=input_price,
            output_cost_per_mtok=output_price,
        )

    def pin(self) -> JudgePin:
        return JudgePin.build(
            provider="anthropic",
            model=self.model,
            rubric_text=self.rubric_text,
            params=self.params,
            scale=self.scale,
        )

    def estimate_cost(self, n_items: int, avg_input_tokens: int = 600) -> CostEstimate:
        return estimate_cost_for(self.describe(), n_items, avg_input_tokens, avg_output_tokens=8)

    def score(self, requests: Sequence[JudgeRequest]) -> Sequence[JudgeResult]:
        client = self._ensure_client()
        results: list[JudgeResult] = []
        for request in requests:
            prompt = self._prompt(request)
            message = client.messages.create(
                model=self.model,
                system=self.rubric_text,
                messages=[{"role": "user", "content": prompt}],
                **self.params,
            )
            text = "".join(
                block.text for block in message.content if getattr(block, "type", "") == "text"
            )
            usage = getattr(message, "usage", None)
            in_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            self.calls += 1
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            results.append(
                JudgeResult(
                    item_id=request.item_id,
                    raw_score=parse_score(text, self.scale),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
            )
        return results

    # ---- plumbing ---------------------------------------------------------------------

    def _prompt(self, request: JudgeRequest) -> str:
        body = f"<input>\n{request.prompt_input}\n</input>\n\n<answer>\n{request.output}\n</answer>"
        if request.nonce:
            body += self.nonce_template.format(nonce=request.nonce)
        return body

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise JudgeCallError(
                "no Anthropic API key found",
                "set ANTHROPIC_API_KEY, or use `--simulate` for the deterministic built-in "
                "judge, which needs no key and no network",
            )
        try:
            import anthropic
        except ImportError as exc:
            raise JudgeCallError(
                "the `anthropic` package is not installed",
                "install the optional extra: `uv pip install 'benchlock[anthropic]'`",
            ) from exc
        self._client = anthropic.Anthropic(api_key=key)
        return self._client
