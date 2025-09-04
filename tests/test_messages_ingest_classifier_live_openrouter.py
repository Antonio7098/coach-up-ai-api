import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

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


def test_mock_openrouter_user_then_assistant_closing(monkeypatch: pytest.MonkeyPatch):
    """
    Use a mocked OpenRouter classifier.
    Flow:
    - user message should typically start/one_off an interaction
    - assistant closing cue should end and enqueue assessment (heuristic ensures end if classifier abstains)
    """
    # Enable classifier and select OpenRouter
    monkeypatch.setenv("AI_CLASSIFIER_ENABLED", "1")
    monkeypatch.setenv("AI_PROVIDER_CLASSIFIER", "openrouter")
    # Optionally pin a model known to support JSON outputs
    monkeypatch.setenv("AI_CLASSIFIER_MODEL", "openai/gpt-4o-mini")
    
    # Mock the OpenRouter classifier to return predictable responses
    with patch('app.providers.factory.get_classifier_client') as mock_get_client:
        mock_client = MagicMock()
        # Mock classify to return start for user message, end for assistant closing
        async def mock_classify(role, content, turn_count):
            if role == "user":
                return {"decision": "start", "confidence": 0.8}
            elif role == "assistant" and "good luck" in content.lower():
                return {"decision": "end", "confidence": 0.9}
            else:
                return {"decision": "continue", "confidence": 0.7}
        mock_client.classify = mock_classify
        mock_get_client.return_value = mock_client

    session_id = "sess_live_openrouter_1"

    with TestClient(app) as client:
        r1 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m1",
                "role": "user",
                "content": "Please plan my next two weeks in a concise bullet list.",
            },
        )
        assert r1.status_code == 200
        j1 = r1.json()
        gid = j1.get("groupId")
        assert isinstance(gid, str) and gid
        assert j1.get("state") in {"active", "idle"}

        # If already ended as one_off, verify enqueue and finish.
        if j1.get("state") == "idle":
            assert j1.get("enqueued") is True
            assert j1.get("turnCount") in (1, 0, None)
            return

        # Otherwise, send assistant closing cue -> should end + enqueue.
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


def test_mock_openrouter_likely_one_off_or_end(monkeypatch: pytest.MonkeyPatch):
    """
    Try a prompt that often gets classified as one_off. If not, still verify
    the pipeline can end via assistant closing and enqueue an assessment.
    """
    monkeypatch.setenv("AI_CLASSIFIER_ENABLED", "1")
    monkeypatch.setenv("AI_PROVIDER_CLASSIFIER", "openrouter")
    monkeypatch.setenv("AI_CLASSIFIER_MODEL", "openai/gpt-4o-mini")
    
    # Mock the OpenRouter classifier to return predictable responses
    with patch('app.providers.factory.get_classifier_client') as mock_get_client:
        mock_client = MagicMock()
        # Mock classify to return one_off for the specific test message
        async def mock_classify(role, content, turn_count):
            if role == "user" and "assess this single message" in content.lower():
                return {"decision": "one_off", "confidence": 0.9}
            elif role == "assistant" and "good luck" in content.lower():
                return {"decision": "end", "confidence": 0.9}
            else:
                return {"decision": "continue", "confidence": 0.7}
        mock_client.classify = mock_classify
        mock_get_client.return_value = mock_client

    session_id = "sess_live_openrouter_2"

    with TestClient(app) as client:
        r = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m1",
                "role": "user",
                "content": "Assess this single message, please.",
            },
        )
        assert r.status_code == 200
        j = r.json()
        gid = j.get("groupId")
        assert isinstance(gid, str) and gid
        assert j.get("state") in {"idle", "active"}

        if j.get("state") == "idle":
            # likely one_off path
            assert j.get("enqueued") is True
            assert j.get("turnCount") in (1, 0, None)
        else:
            # If active, end via assistant closing
            r2 = client.post(
                "/messages/ingest",
                json={
                    "sessionId": session_id,
                    "messageId": "m2",
                    "role": "assistant",
                    "content": "Here's your plan. Good luck!",
                },
            )
            assert r2.status_code == 200
            j2 = r2.json()
            assert j2.get("state") == "idle"
            assert j2.get("enqueued") is True
            assert j2.get("groupId") == gid
