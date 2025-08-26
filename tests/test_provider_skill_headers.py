import json
import hashlib
import types

import pytest

from app.providers.openrouter import OpenRouterAssessClient
import app.providers.openrouter as openrouter_mod


@pytest.mark.asyncio
async def test_openrouter_assess_sets_skill_headers(monkeypatch: pytest.MonkeyPatch):
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
            # minimal valid JSON content response
            payload = {
                "rubricVersion": "v2",
                "categories": ["clarity"],
                "scores": {"clarity": 0.5},
                "highlights": ["h"],
                "recommendations": ["r"],
                "meta": {},
            }
            msg = {"choices": [{"message": {"content": jsonlib.dumps(payload)}}]}
            return FakeResponse(status_code=200, json_data=msg)

    jsonlib = json
    monkeypatch.setattr(openrouter_mod, "httpx", types.SimpleNamespace(AsyncClient=FakeAsyncClient))

    cli = OpenRouterAssessClient(model="x")
    skill = {"id": "skill-42", "name": "Empathy", "category": "soft"}
    out = await cli.assess([{"role": "user", "content": "hi"}], rubric_version="v2", request_id="rid-2", skill=skill)
    assert out.get("rubricVersion") == "v2"

    headers = captured.get("headers") or {}
    assert headers.get("X-Tracked-Skill-Id") == "skill-42"
    expected_hash = hashlib.sha256(b"skill-42").hexdigest()
    assert headers.get("X-Tracked-Skill-Id-Hash") == expected_hash
