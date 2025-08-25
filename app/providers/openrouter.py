import os
from typing import AsyncIterator, Optional

from .base import ChatClient, ClassifierClient, AssessClient


class OpenRouterChatClient(ChatClient):
    provider_name: str = "openrouter"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_CHAT_MODEL") or "openai/gpt-4o-mini")
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter provider")
        # TODO(SPR-005): initialize OpenRouter client (HTTP with proper headers)
        self._api_key = api_key

    async def stream_chat(self, prompt: str, system: Optional[str] = None) -> AsyncIterator[str]:
        # TODO(SPR-005): implement real streaming via OpenRouter SSE/stream API
        # For now, raise to force fallback so we don't silently simulate a real provider
        raise NotImplementedError("OpenRouter provider streaming not implemented yet")


class OpenRouterClassifierClient(ClassifierClient):
    provider_name: str = "openrouter"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_CLASSIFIER_MODEL") or "openai/gpt-4o-mini")
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter provider")
        self._api_key = api_key

    async def classify(self, role: str, content: str, turn_count: int) -> dict:
        # TODO(SPR-005): implement real OpenRouter classification call
        raise NotImplementedError("OpenRouter classifier not implemented yet")


class OpenRouterAssessClient(AssessClient):
    provider_name: str = "openrouter"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_ASSESS_MODEL") or "openai/gpt-4o-mini")
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter provider")
        self._api_key = api_key

    async def assess(self, transcript: list[dict], rubric_version: str = "v1") -> dict:
        # TODO(SPR-005): implement real OpenRouter assessment call
        raise NotImplementedError("OpenRouter assess not implemented yet")
