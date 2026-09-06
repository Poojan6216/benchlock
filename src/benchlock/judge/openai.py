"""The OpenAI judge adapter. Same four methods, same contract as the Anthropic one."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from benchlock.judge.anthropic import JudgeCallError, parse_score
from benchlock.judge.base import (
    CostEstimate,
    JudgeDescription,
    JudgeRequest,
    JudgeResult,
    estimate_cost_for,
)
from benchlock.model.pins import JudgePin

PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
}


def _price_for(model: str) -> tuple[float, float]:
    for prefix, price in PRICES.items():
        if model.startswith(prefix):
            return price
    return (2.50, 10.00)


@dataclass
class OpenAIJudge:
    model: str = "gpt-4o-mini"
    rubric_text: str = ""
    scale: tuple[float, float] = (1.0, 5.0)
    params: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0, "max_tokens": 16})
    api_key: str | None = None
    nonce_template: str = "\n\n<!-- request-id: {nonce} -->"
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    refusals: int = 0  #: policy declines; billed, never scored

    def __post_init__(self) -> None:
        self._client: Any = None

    def describe(self) -> JudgeDescription:
        input_price, output_price = _price_for(self.model)
        return JudgeDescription(
            provider="openai",
            model=self.model,
            scale=self.scale,
            # OpenAI's dated snapshots look like `gpt-4o-2024-08-06`.
            supports_pinned_snapshots=bool(re.search(r"-\d{4}-\d{2}-\d{2}$", self.model)),
            input_cost_per_mtok=input_price,
            output_cost_per_mtok=output_price,
        )

    def pin(self) -> JudgePin:
        return JudgePin.build(
            provider="openai",
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
            body = (
                f"<input>\n{request.prompt_input}\n</input>\n\n"
                f"<answer>\n{request.output}\n</answer>"
            )
            if request.nonce:
                body += self.nonce_template.format(nonce=request.nonce)
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.rubric_text},
                    {"role": "user", "content": body},
                ],
                **self.params,
            )
            # Account for the call before anything can raise: a refused call still bills.
            usage = getattr(completion, "usage", None)
            in_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self.calls += 1
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            choice = completion.choices[0].message
            refusal = getattr(choice, "refusal", None)
            if refusal:
                self.refusals += 1
                raise JudgeCallError(
                    f"the judge declined to score item {request.item_id!r}: {refusal}",
                    "this item's content tripped a policy check; exclude it from the pool "
                    "rather than recording a score that was never produced",
                )
            text = choice.content or ""
            results.append(
                JudgeResult(
                    item_id=request.item_id,
                    raw_score=parse_score(text, self.scale),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
            )
        return results

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise JudgeCallError(
                "no OpenAI API key found",
                "set OPENAI_API_KEY, or use `--simulate` for the deterministic built-in judge",
            )
        try:
            import openai
        except ImportError as exc:
            raise JudgeCallError(
                "the `openai` package is not installed",
                "install the optional extra: `uv pip install 'benchlock[openai]'`",
            ) from exc
        self._client = openai.OpenAI(api_key=key)
        return self._client
