from __future__ import annotations

import abc
from typing import AsyncIterator, Optional


class ChatClient(abc.ABC):
    """Abstract chat client interface supporting streaming.

    Implementations should yield string tokens (without SSE framing).
    """

    provider_name: str = "unknown"

    def __init__(self, model: Optional[str] = None):
        self.model = model

    @abc.abstractmethod
    async def stream_chat(self, prompt: str, system: Optional[str] = None) -> AsyncIterator[str]:
        ...


class ClassifierClient(abc.ABC):
    """Placeholder interface for boundary classification."""

    provider_name: str = "unknown"

    def __init__(self, model: Optional[str] = None):
        self.model = model

    @abc.abstractmethod
    async def classify(self, role: str, content: str, turn_count: int) -> dict:
        ...


class AssessClient(abc.ABC):
    """Placeholder interface for assessments."""

    provider_name: str = "unknown"

    def __init__(self, model: Optional[str] = None):
        self.model = model

    @abc.abstractmethod
    async def assess(self, transcript: list[dict], rubric_version: str = "v1") -> dict:
        ...
