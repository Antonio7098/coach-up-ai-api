import json
import asyncio
import os
import types

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.main import app
from app.providers.openrouter import (
    OpenRouterChatClient,
    OpenRouterClassifierClient,
    OpenRouterAssessClient,
)
import app.providers.openrouter as openrouter_mod


@pytest.mark.asyncio
async def test_openrouter_classify_propagates_x_request_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    captured = {}

    class FakeResponse:
        def __init__(self, status_code=200, json_data=None, text=""):
            self.status_code = status_code
            self._json = json_data or {}
            self.text = text

        def json(self):
            return self._json

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            captured["headers"] = headers or {}
            # Return a minimal but valid OpenAI-style response payload
            msg = {
                "choices": [
                    {"message": {"content": jsonlib.dumps({"decision": "start", "confidence": 0.9, "reasons": "ok"})}}
                ]
            }
            return FakeResponse(status_code=200, json_data=msg)

    # Provide jsonlib inside closure for the FakeAsyncClient
    jsonlib = json
    monkeypatch.setattr(openrouter_mod, "httpx", types.SimpleNamespace(AsyncClient=FakeAsyncClient))

    cli = OpenRouterClassifierClient(model="x")
    out = await cli.classify("user", "hi", 1, request_id="req-123")
    assert out["decision"] in {"start", "abstain"}
    assert captured["headers"].get("X-Request-Id") == "req-123"


@pytest.mark.asyncio
async def test_openrouter_assess_propagates_x_request_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    captured = {}

    class FakeResponse:
        def __init__(self, status_code=200, json_data=None, text=""):
            self.status_code = status_code
            self._json = json_data or {}
            self.text = text

        def json(self):
            return self._json

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            captured["headers"] = headers or {}
            # Return a minimal JSON chat response with strict JSON content
            payload = {
                "rubricVersion": "v2",
                "categories": ["clarity"],
                "scores": {"clarity": 0.8},
                "highlights": ["good"],
                "recommendations": ["do more"],
                "meta": {"ok": True},
            }
            msg = {"choices": [{"message": {"content": jsonlib.dumps(payload)}}]}
            return FakeResponse(status_code=200, json_data=msg)

    jsonlib = json
    import types as _types

    monkeypatch.setattr(openrouter_mod, "httpx", _types.SimpleNamespace(AsyncClient=FakeAsyncClient))

    cli = OpenRouterAssessClient(model="x")
    out = await cli.assess([{"role": "user", "content": "hi"}], rubric_version="v2", request_id="req-xyz")
    assert isinstance(out, dict) and out.get("rubricVersion")
    assert captured["headers"].get("X-Request-Id") == "req-xyz"


@pytest.mark.asyncio
async def test_openrouter_chat_stream_propagates_x_request_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    captured = {}

    class FakeStreamResp:
        def __init__(self):
            self.status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            # Yield one token then done
            chunk = {"choices": [{"delta": {"content": "Hello"}}]}
            yield "data: " + json.dumps(chunk)
            yield "data: [DONE]"

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers=None, json=None):
            captured["headers"] = headers or {}
            return FakeStreamResp()

    import types as _types

    monkeypatch.setattr(openrouter_mod, "httpx", _types.SimpleNamespace(AsyncClient=FakeAsyncClient))

    cli = OpenRouterChatClient(model="x")
    tokens = []
    async for tok in cli.stream_chat("hi", request_id="rid-1"):
        tokens.append(tok)
    assert "Hello" in "".join(tokens)
    assert captured["headers"].get("X-Request-Id") == "rid-1"


def _get_metric_count(name: str, labels: dict) -> float:
    val = REGISTRY.get_sample_value(name, labels)
    return float(val) if val is not None else 0.0


def test_chat_metrics_emitted_on_fallback_stream(monkeypatch: pytest.MonkeyPatch):
    # Ensure provider path disabled so fallback/stub path is used
    monkeypatch.delenv("AI_CHAT_ENABLED", raising=False)

    before_ttft = _get_metric_count(
        "coachup_chat_ttft_seconds_count", {"provider": "stub", "model": "unknown"}
    )
    before_total = _get_metric_count(
        "coachup_chat_total_seconds_count", {"provider": "stub", "model": "unknown"}
    )

    with TestClient(app) as client:
        with client.stream("GET", "/chat/stream") as resp:
            # Read just one line to trigger first-token instrumentation
            for line in resp.iter_lines():
                if line:
                    break
        # Closing the stream should finalize and record total seconds

    after_ttft = _get_metric_count(
        "coachup_chat_ttft_seconds_count", {"provider": "stub", "model": "unknown"}
    )
    after_total = _get_metric_count(
        "coachup_chat_total_seconds_count", {"provider": "stub", "model": "unknown"}
    )

    assert after_ttft >= before_ttft + 1
    assert after_total >= before_total + 1
