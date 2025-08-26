import asyncio
import json
import time
from typing import Dict, Any, List

import pytest
from fastapi.testclient import TestClient

import app.main as main


class _FakeHTTPResponse:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def read(self):
        return b"{}"


def _wait_for(predicate, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_persist_v2_payload_shape(monkeypatch: pytest.MonkeyPatch):
    # Enable assess flow, set flag to v2 and provide tracked skills
    monkeypatch.setenv("AI_ASSESS_ENABLED", "1")
    # ASSESS_OUTPUT_V2 deprecated; v2 is always used
    skills = [
        {"id": "s1", "name": "Clarity", "category": "communication"},
        {"id": "s2", "name": "Empathy", "category": "soft"},
    ]
    monkeypatch.setenv("ASSESS_TRACKED_SKILLS_JSON", json.dumps(skills))

    # Fake assess client returns per-skill distinct outputs
    class FakeAssessClient:
        provider_name = "test"
        model = "m"
        async def assess(self, transcript: List[Dict[str, Any]], rubric_version: str = "v2", request_id: str | None = None, skill: Dict[str, Any] | None = None) -> Dict[str, Any]:
            sid = (skill or {}).get("id")
            score = 0.2 if sid == "s1" else 0.8
            return {
                "rubricVersion": rubric_version,
                "categories": ["clarity"],
                "scores": {"clarity": score},
                "highlights": [f"h-{sid}"],
                "recommendations": [f"r-{sid}"],
                "meta": {"provider": "test", "skill": {k: (skill or {}).get(k) for k in ("id", "name", "category")}},
            }

    monkeypatch.setattr(main, "get_assess_client", lambda: FakeAssessClient())

    captured: Dict[str, Any] = {}
    def fake_urlopen(req, timeout):
        captured["url"] = getattr(req, "full_url", None)
        captured["headers"] = dict(req.header_items())
        captured["data"] = req.data
        captured["method"] = req.get_method()
        return _FakeHTTPResponse()

    monkeypatch.setattr(main.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(main, "PERSIST_ASSESSMENTS_URL", "http://example.com/finalize")
    monkeypatch.setattr(main, "PERSIST_ASSESSMENTS_SECRET", "secret")

    # Drive via API to exercise worker and persistence
    with TestClient(main.app) as client:
        r = client.post("/assessments/run", json={"sessionId": "sess-v2"})
        assert r.status_code == 200
        group_id = r.json()["groupId"]
        ok = _wait_for(lambda: bool(captured.get("data")), timeout_s=5.0)
        assert ok, "persistence request not observed"

    # Validate payload
    body = json.loads((captured.get("data") or b"{}").decode("utf-8"))
    assert body.get("sessionId") == "sess-v2"
    assert body.get("groupId") == group_id
    assert body.get("rubricVersion") == "v2"

    summary = body.get("summary") or {}
    # v2 shape: { skillAssessments: [...], meta: {...} }
    sa = summary.get("skillAssessments") or []
    assert isinstance(sa, list) and len(sa) == 2
    # Items contain skillHash, not raw skill id objects
    for item in sa:
        assert "skillHash" in item and isinstance(item["skillHash"], (str, type(None)))
        assert "metCriteria" in item and isinstance(item["metCriteria"], list)
        assert "unmetCriteria" in item and isinstance(item["unmetCriteria"], list)
        assert "feedback" in item and isinstance(item["feedback"], list)
        # Ensure we did not leak raw skill identifiers in v2 payload
        assert "skill" not in item

    meta = summary.get("meta") or {}
    assert meta.get("provider") in ("test", None)
    # model may be present or omitted depending on provider test wiring
    assert meta.get("skillsCount") == 2
