import os
from typing import AsyncIterator, Optional

from .base import ChatClient, ClassifierClient, AssessClient


class GoogleChatClient(ChatClient):
    provider_name: str = "google"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_CHAT_MODEL") or "gemini-1.5-pro")
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Google provider")
        # TODO(SPR-005): initialize google generative ai client with api_key
        self._api_key = api_key

    async def stream_chat(self, prompt: str, system: Optional[str] = None, request_id: Optional[str] = None) -> AsyncIterator[str]:
        # TODO(SPR-005): implement real streaming via Google Generative AI SDK
        # For now, raise to force fallback so we don't silently simulate a real provider
        raise NotImplementedError("Google provider streaming not implemented yet")


class GoogleClassifierClient(ClassifierClient):
    provider_name: str = "google"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_CLASSIFIER_MODEL") or "gemini-1.5-pro")
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Google provider")
        self._api_key = api_key

    async def classify(self, role: str, content: str, turn_count: int, request_id: Optional[str] = None) -> dict:
        # TODO(SPR-005): implement real Google classification
        # Raise to let caller fall back to mock/heuristics during early scaffolding
        raise NotImplementedError("Google classifier not implemented yet")


class GoogleAssessClient(AssessClient):
    provider_name: str = "google"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_ASSESS_MODEL") or "gemini-1.5-pro")
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Google provider")
        self._api_key = api_key

    async def assess(self, transcript: list[dict], rubric_version: str = "v1", request_id: Optional[str] = None, skill: Optional[dict] = None) -> dict:
        # TODO(SPR-005): implement real Google assessment client
        raise NotImplementedError("Google assess not implemented yet")
