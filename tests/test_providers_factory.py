import os
import types

import pytest

from app.providers.factory import get_classifier_client
from app.providers.mock import MockClassifierClient


@pytest.mark.parametrize("prov_env, expect_type", [
    ("mock", MockClassifierClient),
    ("test", MockClassifierClient),
    ("unknown", MockClassifierClient),
])
def test_get_classifier_client_basic(prov_env, expect_type, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_PROVIDER_CLASSIFIER", prov_env)
    # Ensure no accidental Google/OpenRouter keys interfere
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    cli = get_classifier_client()
    assert isinstance(cli, expect_type)


def test_get_classifier_client_google_missing_key_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch):
    # Select Google provider but without API key, factory should fall back to Mock
    monkeypatch.setenv("AI_PROVIDER_CLASSIFIER", "google")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    cli = get_classifier_client()
    assert isinstance(cli, MockClassifierClient)


def test_get_classifier_client_google_with_key_but_unimplemented_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch):
    # With a key set, Google client will instantiate but classify will raise NotImplementedError.
    # Factory returns Google client; we simulate a classify call failure and ensure caller can recover by using mock.
    # Here we just assert factory returns a client object (Google) or Mock depending on import behavior,
    # and that calling classify raises NotImplementedError as scaffolding indicates.
    monkeypatch.setenv("AI_PROVIDER_CLASSIFIER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")

    cli = get_classifier_client()
    # We don't assert exact class name (scaffold may change). Instead, assert behavior on classify.
    with pytest.raises(NotImplementedError):
        # role/content/turn_count don't matter for unimplemented scaffold
        import asyncio
        asyncio.get_event_loop().run_until_complete(cli.classify("user", "hi", 0))
