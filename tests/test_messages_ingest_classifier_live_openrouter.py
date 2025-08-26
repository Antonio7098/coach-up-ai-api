import os
import pytest
from fastapi.testclient import TestClient

from app.main import app


requires_live = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_OPENROUTER") or not os.getenv("OPENROUTER_API_KEY"),
    reason="Live OpenRouter tests skipped (set RUN_LIVE_OPENROUTER=1 and OPENROUTER_API_KEY)",
)


@pytest.fixture(autouse=True)
def reset_state():
    # Ensure per-test isolation of in-memory state
    app.state.processed_message_ids = set()  # type: ignore[attr-defined]
    app.state.session_state = {}  # type: ignore[attr-defined]
    app.state.session_transcripts = {}  # type: ignore[attr-defined]
    app.state.group_spans = {}  # type: ignore[attr-defined]
    app.state.assessment_results = {}  # type: ignore[attr-defined]
    yield


@requires_live
@pytest.mark.timeout(30)
def test_live_openrouter_user_then_assistant_closing(monkeypatch: pytest.MonkeyPatch):
    """
    Use the real OpenRouter classifier.
    Flow:
    - user message should typically start/one_off an interaction
    - assistant closing cue should end and enqueue assessment (heuristic ensures end if classifier abstains)
    """
    # Enable classifier and select OpenRouter
    monkeypatch.setenv("AI_CLASSIFIER_ENABLED", "1")
    monkeypatch.setenv("AI_PROVIDER_CLASSIFIER", "openrouter")
    # Optionally pin a model known to support JSON outputs
    monkeypatch.setenv("AI_CLASSIFIER_MODEL", os.getenv("AI_CLASSIFIER_MODEL", "openai/gpt-4o-mini"))

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


@requires_live
@pytest.mark.timeout(30)
def test_live_openrouter_likely_one_off_or_end(monkeypatch: pytest.MonkeyPatch):
    """
    Try a prompt that often gets classified as one_off. If not, still verify
    the pipeline can end via assistant closing and enqueue an assessment.
    """
    monkeypatch.setenv("AI_CLASSIFIER_ENABLED", "1")
    monkeypatch.setenv("AI_PROVIDER_CLASSIFIER", "openrouter")
    monkeypatch.setenv("AI_CLASSIFIER_MODEL", os.getenv("AI_CLASSIFIER_MODEL", "openai/gpt-4o-mini"))

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
