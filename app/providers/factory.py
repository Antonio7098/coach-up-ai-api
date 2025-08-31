import os
from typing import Optional

from .base import ChatClient, ClassifierClient, AssessClient, SummaryClient
from .mock import MockChatClient, MockClassifierClient, MockAssessClient, MockSummaryClient


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_chat_client(provider: Optional[str] = None, model: Optional[str] = None) -> ChatClient:
    """Return a chat client based on env or explicit overrides.

    Env precedence:
      - AI_PROVIDER_CHAT
      - AI_PROVIDER
      - defaults to 'mock'
    Model from AI_CHAT_MODEL if not given.
    """
    prov = (provider or _env_str("AI_PROVIDER_CHAT") or _env_str("AI_PROVIDER") or "mock").lower()
    mdl = model or _env_str("AI_CHAT_MODEL") or None

    if prov in ("mock", "test"):
        return MockChatClient(model=mdl)

    if prov in ("google", "gemini"):
        try:
            from .google import GoogleChatClient  # type: ignore
            return GoogleChatClient(model=mdl)
        except Exception:
            # Fallback to mock if provider module/keys not available
            return MockChatClient(model=mdl)

    if prov in ("openrouter", "router"):
        try:
            from .openrouter import OpenRouterChatClient  # type: ignore
            return OpenRouterChatClient(model=mdl)
        except Exception:
            return MockChatClient(model=mdl)

    # Unknown -> mock
    return MockChatClient(model=mdl)


def get_classifier_client(provider: Optional[str] = None, model: Optional[str] = None) -> ClassifierClient:
    prov = (provider or _env_str("AI_PROVIDER_CLASSIFIER") or _env_str("AI_PROVIDER") or "mock").lower()
    mdl = model or _env_str("AI_CLASSIFIER_MODEL") or None

    if prov in ("mock", "test"):
        return MockClassifierClient(model=mdl)

    if prov in ("google", "gemini"):
        try:
            from .google import GoogleClassifierClient  # type: ignore
            return GoogleClassifierClient(model=mdl)
        except Exception:
            return MockClassifierClient(model=mdl)

    if prov in ("openrouter", "router"):
        try:
            from .openrouter import OpenRouterClassifierClient  # type: ignore
            return OpenRouterClassifierClient(model=mdl)
        except Exception:
            return MockClassifierClient(model=mdl)

    return MockClassifierClient(model=mdl)


def get_assess_client(provider: Optional[str] = None, model: Optional[str] = None) -> AssessClient:
    prov = (provider or _env_str("AI_PROVIDER_ASSESS") or _env_str("AI_PROVIDER") or "mock").lower()
    mdl = model or _env_str("AI_ASSESS_MODEL") or None

    if prov in ("mock", "test"):
        return MockAssessClient(model=mdl)

    if prov in ("google", "gemini"):
        try:
            from .google import GoogleAssessClient  # type: ignore
            return GoogleAssessClient(model=mdl)
        except Exception:
            return MockAssessClient(model=mdl)

    if prov in ("openrouter", "router"):
        try:
            from .openrouter import OpenRouterAssessClient  # type: ignore
            return OpenRouterAssessClient(model=mdl)
        except Exception:
            return MockAssessClient(model=mdl)

    return MockAssessClient(model=mdl)


def get_summary_client(provider: Optional[str] = None, model: Optional[str] = None) -> SummaryClient:
    prov = (provider or _env_str("AI_PROVIDER_SUMMARY") or _env_str("AI_PROVIDER") or "mock").lower()
    mdl = model or _env_str("AI_SUMMARY_MODEL") or _env_str("AI_CHAT_MODEL") or None

    if prov in ("mock", "test"):
        return MockSummaryClient(model=mdl)

    if prov in ("google", "gemini"):
        try:
            from .google import GoogleSummaryClient  # type: ignore
            return GoogleSummaryClient(model=mdl)
        except Exception:
            return MockSummaryClient(model=mdl)

    # Fallback
    return MockSummaryClient(model=mdl)
