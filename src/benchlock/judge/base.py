"""The judge provider interface: four methods, and nothing that decides anything.

A judge produces numbers. Everything downstream consumes numbers. This module is the only
place in `benchlock` that talks to a model, and it is deliberately the narrowest possible
surface — no prompting strategy, no retry-and-reinterpret, no "ask the model whether the
score seems right". Hard Rule 1 lives or dies on keeping this boundary thin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from benchlock.model.pins import JudgePin


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    """One item to score. `output` is the system's answer; `prompt_input` is the question."""

    item_id: str
    prompt_input: str
    output: str
    #: Per-run cache-busting nonce (Phase 7.9). Empty when disabled.
    nonce: str = ""


@dataclass(frozen=True, slots=True)
class JudgeResult:
    item_id: str
    raw_score: float
    #: Tokens consumed, for the cost estimate. Zero when the provider does not report them.
    input_tokens: int = 0
    output_tokens: int = 0
    #: True when the provider indicated the response came from a cache (Phase 7.9).
    cached: bool = False


@dataclass(frozen=True, slots=True)
class JudgeDescription:
    provider: str
    model: str
    scale: tuple[float, float]
    #: Whether the provider offers dated snapshots that can be pinned. `replicate` mode
    #: requires this and refuses without it (Phase 3.3).
    supports_pinned_snapshots: bool
    #: Dollars per million input / output tokens, for `benchlock plan`'s cost estimate.
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0


@dataclass(frozen=True, slots=True)
class CostEstimate:
    items: int
    input_tokens: int
    output_tokens: int
    dollars: float

    def to_json(self) -> dict[str, float | int]:
        return {
            "items": self.items,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "dollars": round(self.dollars, 4),
        }


class JudgeAdapter(Protocol):
    """Four methods. Any more and the decision path starts leaking into the provider."""

    def describe(self) -> JudgeDescription: ...

    def pin(self) -> JudgePin: ...

    def score(self, requests: Sequence[JudgeRequest]) -> Sequence[JudgeResult]: ...

    def estimate_cost(self, n_items: int, avg_input_tokens: int = 600) -> CostEstimate: ...


class SimulatedJudge:
    """A deterministic judge with injectable drift. Used by tests and the simulation study.

    It is not a mock in the usual sense: it implements the real interface and produces
    scores through the same path a hosted provider would, so the end-to-end anchor flow is
    exercised for real. What it adds is control over the two things a hosted judge will
    not give you on demand — a known noise level, and a drift you can switch on at a
    chosen run.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        per_item_sd: float = 0.08,
        drift: float = 0.0,
        scale: tuple[float, float] = (1.0, 5.0),
        model: str = "simulated-judge-v1",
        rubric_text: str = "Score the answer 1-5.\n",
        supports_pinned_snapshots: bool = True,
        cache: bool = False,
        bias: float = 0.0,
    ) -> None:
        import numpy as np

        self._rng = np.random.default_rng(seed)
        self._np = np
        self.per_item_sd = per_item_sd
        self.drift = drift
        self.scale = scale
        self.model = model
        self.rubric_text = rubric_text
        self._supports_pinning = supports_pinned_snapshots
        #: Phase 7.9: when set, identical requests return the identical first answer, so
        #: real judge drift becomes invisible unless a nonce is used.
        self.cache = cache
        self._cache: dict[str, float] = {}
        #: A constant offset applied to every score. A judge that is *wrong* but stable.
        self.bias = bias
        self.calls = 0

    def describe(self) -> JudgeDescription:
        return JudgeDescription(
            provider="simulated",
            model=self.model,
            scale=self.scale,
            supports_pinned_snapshots=self._supports_pinning,
            input_cost_per_mtok=3.0,
            output_cost_per_mtok=15.0,
        )

    def pin(self) -> JudgePin:
        return JudgePin.build(
            provider="simulated",
            model=self.model,
            rubric_text=self.rubric_text,
            params={"temperature": 0.0, "max_tokens": 512},
            scale=self.scale,
        )

    def _latent(self, item_id: str) -> float:
        """A stable per-item quality, derived from the id so it never drifts on its own."""
        import hashlib

        digest = hashlib.sha256(item_id.encode()).digest()
        return 0.25 + 0.5 * (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF)

    def score(self, requests: Sequence[JudgeRequest]) -> Sequence[JudgeResult]:
        self.calls += len(requests)
        results = []
        for request in requests:
            key = f"{request.item_id}|{request.output}|{request.nonce}"
            if self.cache and key in self._cache:
                results.append(JudgeResult(request.item_id, self._cache[key], 600, 20, cached=True))
                continue
            unit = self._latent(request.item_id) + self.drift + self.bias
            unit += float(self._rng.normal(0.0, self.per_item_sd))
            unit = min(max(unit, 0.0), 1.0)
            lo, hi = self.scale
            raw = lo + unit * (hi - lo)
            if self.cache:
                self._cache[key] = raw
            results.append(JudgeResult(request.item_id, raw, 600, 20))
        return results

    def estimate_cost(self, n_items: int, avg_input_tokens: int = 600) -> CostEstimate:
        description = self.describe()
        input_tokens = n_items * avg_input_tokens
        output_tokens = n_items * 20
        dollars = (
            input_tokens / 1e6 * description.input_cost_per_mtok
            + output_tokens / 1e6 * description.output_cost_per_mtok
        )
        return CostEstimate(n_items, input_tokens, output_tokens, dollars)


def estimate_cost_for(
    description: JudgeDescription,
    n_items: int,
    avg_input_tokens: int = 600,
    avg_output_tokens: int = 20,
) -> CostEstimate:
    """Shared cost arithmetic, so every adapter reports it the same way."""
    input_tokens = n_items * avg_input_tokens
    output_tokens = n_items * avg_output_tokens
    dollars = (
        input_tokens / 1e6 * description.input_cost_per_mtok
        + output_tokens / 1e6 * description.output_cost_per_mtok
    )
    return CostEstimate(n_items, input_tokens, output_tokens, dollars)
