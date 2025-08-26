import asyncio
from typing import AsyncIterator, Optional, Dict, Any, List

from .base import ChatClient, ClassifierClient, AssessClient


class MockChatClient(ChatClient):
    provider_name: str = "mock"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or "mock-chat-1")

    async def stream_chat(self, prompt: str, system: Optional[str] = None, request_id: Optional[str] = None) -> AsyncIterator[str]:
        text = prompt or "Hello, world!"
        # Stream in small chunks deterministically
        for token in _chunk_text(text, size=6):
            yield token
            await asyncio.sleep(0.05)


def _chunk_text(s: str, size: int = 6):
    for i in range(0, len(s), size):
        yield s[i : i + size]


class MockClassifierClient(ClassifierClient):
    provider_name: str = "mock"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or "mock-cls-1")

    async def classify(self, role: str, content: str, turn_count: int, request_id: Optional[str] = None) -> Dict[str, Any]:
        text = (content or "").lower()
        # Simple deterministic pattern to emulate boundary decisions
        if role == "assistant" and any(cue in text for cue in ["good luck", "let me know", "anything else", "does that help", "glad to help"]):
            return {"decision": "end", "confidence": 0.75, "reasons": "closing_cue"}
        if role == "user" and any(cue in text for cue in ["plan", "over the next", "for two weeks", "step-by-step"]):
            return {"decision": "start" if turn_count == 0 else "continue", "confidence": 0.68, "reasons": "planning_intent"}
        # default low confidence abstain
        return {"decision": "abstain", "confidence": 0.2, "reasons": "unclear"}


class MockAssessClient(AssessClient):
    provider_name: str = "mock"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or "mock-assess-1")

    async def assess(
        self,
        transcript: List[Dict[str, Any]],
        rubric_version: str = "v1",
        request_id: Optional[str] = None,
        skill: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Deterministic placeholder including optional skill context in meta
        meta: Dict[str, Any] = {"provider": self.provider_name}
        if request_id:
            meta["requestId"] = request_id
        if skill:
            meta["skill"] = {k: skill.get(k) for k in ("id", "name", "category") if k in skill}
        return {
            "rubricVersion": rubric_version,
            "categories": [],
            "scores": {},
            "highlights": ["placeholder"],
            "recommendations": ["placeholder"],
            "meta": meta,
        }
