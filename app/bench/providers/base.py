from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Protocol, TypedDict


class GenResult(TypedDict):
    output: str
    tokensIn: int
    tokensOut: int
    latencyMs: int
    provider: str
    modelId: str


@dataclass(frozen=True)
class RunConfig:
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 300
    timeoutMs: int = 15000
    retries: int = 1


class ChatProvider(Protocol):
    provider: str
    modelId: str

    def generate(self, messages: list[dict], config: RunConfig) -> GenResult:  # sync for simplicity
        ...


def _estimate_tokens(text: str) -> int:
    # Very rough heuristic to avoid external deps
    # ~1 token per 4 chars as a crude approximation
    return max(1, int(len(text) / 4))


class MockProvider:
    """Deterministic provider for harness validation.

    Behaviors:
    - Echoes minimal rewrite for micro_correction/word_choice
    - Adds a short, assertive tweak for assertiveness
    - Summarizes by truncation for summary
    - Returns fixed classification JSON for interaction_classification
    - Returns a structured assessment stub for assessment
    """

    def __init__(self, modelId: str = "mock-1") -> None:
        self.provider = "mock"
        self.modelId = modelId

    def generate(self, messages: list[dict], config: RunConfig) -> GenResult:
        start = time.perf_counter()
        user_texts = [m["content"] for m in messages if m.get("role") == "user"]
        last = user_texts[-1] if user_texts else ""
        out = last.strip()
        # Minimal edits: fix a couple of obvious patterns for demo purposes
        out = out.replace(" i ", " I ")
        out = out.replace(" i'", " I'")
        out = out.replace(" i\'", " I\'")
        out = out.replace(" i,", " I,")
        out = out.replace(" i.", " I.")
        out = out.replace(" i ", " I ")
        out = out.replace(" i go ", " I went ")
        if "revert" in out.lower():
            out = out.replace("revert", "reply")
        elapsed = int((time.perf_counter() - start) * 1000)
        text_in = "\n".join([m.get("content", "") for m in messages])
        return GenResult(
            output=out,
            tokensIn=_estimate_tokens(text_in),
            tokensOut=_estimate_tokens(out),
            latencyMs=max(5, elapsed),
            provider=self.provider,
            modelId=self.modelId,
        )
