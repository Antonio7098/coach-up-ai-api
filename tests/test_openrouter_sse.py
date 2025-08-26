import json
import types
import pytest

import app.providers.openrouter as openrouter_mod
from app.providers.openrouter import OpenRouterChatClient


@pytest.mark.asyncio
async def test_openrouter_sse_parsing_skips_malformed_and_yields_tokens(monkeypatch: pytest.MonkeyPatch):
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    class FakeStreamResp:
        def __init__(self):
            self.status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            # Empty line -> ignored
            yield ""
            # Comment/keepalive -> ignored
            yield ": keepalive"
            # Non-data event -> ignored
            yield "event: meta"
            # Malformed JSON -> ignored
            yield "data: {bad json"
            # Empty data -> ignored
            yield "data:   "
            # Valid chunk with text field fallback
            chunk1 = {"choices": [{"text": "Hello "}]}
            yield "data: " + json.dumps(chunk1)
            # Valid chunk with delta.content
            chunk2 = {"choices": [{"delta": {"content": "world"}}]}
            yield "data: " + json.dumps(chunk2)
            # Done
            yield "data: [DONE]"

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers=None, json=None):
            return FakeStreamResp()

    monkeypatch.setattr(openrouter_mod, "httpx", types.SimpleNamespace(AsyncClient=FakeAsyncClient))

    cli = OpenRouterChatClient(model="unit-test-model")

    # Act
    tokens = []
    async for t in cli.stream_chat("hi", system=None, request_id="rid-test"):
        tokens.append(t)

    # Assert
    assert "".join(tokens) == "Hello world"


@pytest.mark.asyncio
async def test_openrouter_sse_4xx_raises_runtimeerror(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    class FakeStreamResp:
        def __init__(self):
            self.status_code = 429

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return b"rate limited"

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers=None, json=None):
            return FakeStreamResp()

    monkeypatch.setattr(openrouter_mod, "httpx", types.SimpleNamespace(AsyncClient=FakeAsyncClient))

    cli = OpenRouterChatClient(model="unit-test-model")

    with pytest.raises(RuntimeError) as ei:
        async for _ in cli.stream_chat("hi"):
            pass
    assert "OpenRouter error 429" in str(ei.value)
