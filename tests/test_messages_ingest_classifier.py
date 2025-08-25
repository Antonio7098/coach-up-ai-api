import asyncio
import os

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app


@pytest.fixture(autouse=True)
def reset_state():
    # Ensure per-test isolation of in-memory state
    app.state.processed_message_ids = set()  # type: ignore[attr-defined]
    app.state.session_state = {}  # type: ignore[attr-defined]
    app.state.session_transcripts = {}  # type: ignore[attr-defined]
    app.state.group_spans = {}  # type: ignore[attr-defined]
    app.state.assessment_results = {}  # type: ignore[attr-defined]
    yield


def test_ingest_provider_enabled_with_mock_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_CLASSIFIER_ENABLED", "1")
    monkeypatch.setenv("AI_PROVIDER_CLASSIFIER", "mock")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    session_id = "sess_cls_mock"
    with TestClient(app) as client:
        r1 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m1",
                "role": "user",
                "content": "Can you plan my next two weeks?",
            },
        )
        assert r1.status_code == 200
        j1 = r1.json()
        assert j1.get("state") == "active"
        assert j1.get("turnCount") == 1
        assert not j1.get("enqueued")
        gid = j1.get("groupId")
        assert isinstance(gid, str) and gid

        r2 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m2",
                "role": "assistant",
                "content": "Here is your plan. Good luck!",
            },
        )
        assert r2.status_code == 200
        j2 = r2.json()
        assert j2.get("state") == "idle"
        assert j2.get("enqueued") is True
        assert j2.get("groupId") == gid


def test_ingest_provider_enabled_google_classify_raises_and_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch):
    # Enable provider path and select google; classification should raise NotImplemented and fall back to mock
    monkeypatch.setenv("AI_CLASSIFIER_ENABLED", "1")
    monkeypatch.setenv("AI_PROVIDER_CLASSIFIER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")

    session_id = "sess_cls_google_fallback"
    with TestClient(app) as client:
        r1 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m1",
                "role": "user",
                "content": "Please plan the next two weeks",
            },
        )
        assert r1.status_code == 200
        j1 = r1.json()
        # Should behave like mock classification fallback: start + active
        assert j1.get("state") == "active"
        assert j1.get("turnCount") == 1


def test_ingest_provider_enabled_complete_failure_uses_stub(monkeypatch: pytest.MonkeyPatch):
    # Force both provider selection and explicit mock selection to fail -> fallback to local stub
    def raising_get_client(*args, **kwargs):
        raise RuntimeError("factory failure")

    monkeypatch.setenv("AI_CLASSIFIER_ENABLED", "1")
    monkeypatch.setenv("AI_PROVIDER_CLASSIFIER", "mock")
    monkeypatch.setattr(main, "get_classifier_client", raising_get_client)

    session_id = "sess_cls_stub_fallback"
    with TestClient(app) as client:
        r1 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m1",
                "role": "user",
                "content": "Can you plan for two weeks?",
            },
        )
        assert r1.status_code == 200
        j1 = r1.json()
        # Stub returns decision start with conf 0.65 (>= low, < accept), thus kept as start
        assert j1.get("state") == "active"
        assert j1.get("turnCount") == 1


def test_ingest_provider_enabled_low_confidence_override_via_heuristic(monkeypatch: pytest.MonkeyPatch):
    # Provide a fake client that returns low-confidence 'ignore' to ensure heuristics override to 'start' for idle+user
    class FakeLowConfClient:
        provider_name = "fake"
        model = "fake-1"

        async def classify(self, role: str, content: str, turn_count: int):
            return {"decision": "ignore", "confidence": 0.2}

    def fake_get_client(*args, **kwargs):
        return FakeLowConfClient()

    monkeypatch.setenv("AI_CLASSIFIER_ENABLED", "1")
    monkeypatch.setattr(main, "get_classifier_client", fake_get_client)

    session_id = "sess_cls_low_conf"
    with TestClient(app) as client:
        r1 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m1",
                "role": "user",
                "content": "Hello",
            },
        )
        assert r1.status_code == 200
        j1 = r1.json()
        # Heuristic for idle user is start
        assert j1.get("state") == "active"
        assert j1.get("turnCount") == 1
