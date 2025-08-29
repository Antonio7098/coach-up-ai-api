import time
from typing import Dict

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app


def _poll_summary(client: TestClient, session_id: str, expect_group: str, timeout_s: float = 5.0) -> Dict:
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        r = client.get(f"/assessments/{session_id}")
        assert r.status_code == 200
        last = r.json()
        # Return as soon as the expected group is reflected; scores may be empty if providers failed
        if last.get("latestGroupId") == expect_group and last.get("summary") is not None:
            return last
        time.sleep(0.05)
    return last


def test_messages_ingest_start_end_flow_and_dedup():
    session_id = "sess_ingest_1"
    with TestClient(app) as client:
        # Start: user message with planning cue → LLM stub returns start (conf 0.65), accepted False but not low → keep "start"
        m1 = {
            "sessionId": session_id,
            "messageId": "m_u_1",
            "role": "user",
            "content": "Can you plan my next two weeks?",
        }
        r1 = client.post("/messages/ingest", json=m1)
        assert r1.status_code == 200, r1.text
        j1 = r1.json()
        assert j1.get("state") == "active"
        assert isinstance(j1.get("groupId"), str) and j1["groupId"]
        assert j1.get("turnCount") == 1
        assert j1.get("enqueued") is False
        group_id = j1["groupId"]

        # Idempotency: same messageId should be deduped and not enqueue
        r1b = client.post("/messages/ingest", json=m1)
        assert r1b.status_code == 200
        j1b = r1b.json()
        assert j1b.get("deduped") is True
        assert j1b.get("enqueued") is False
        assert j1b.get("groupId") == group_id

        # End: assistant closing cue → LLM stub returns end (conf 0.72) → enqueue assessment
        m2 = {
            "sessionId": session_id,
            "messageId": "m_a_2",
            "role": "assistant",
            "content": "Here is your plan. Good luck!",
        }
        r2 = client.post("/messages/ingest", json=m2)
        assert r2.status_code == 200, r2.text
        j2 = r2.json()
        assert j2.get("state") == "idle"
        assert j2.get("enqueued") is True
        assert j2.get("groupId") == group_id

        # Worker should produce scores and persist to in-memory results
        payload = _poll_summary(client, session_id, group_id)
        assert payload.get("sessionId") == session_id
        assert payload.get("latestGroupId") == group_id
        summary = payload.get("summary", {})
        scores = summary.get("scores", {})
        job_err = summary.get("jobError")
        # Accept either non-empty scores or an explicit jobError after retries were exhausted
        assert (isinstance(scores, dict) and len(scores) > 0) or job_err == "empty_scores_retries_exhausted"


def test_messages_ingest_missing_fields_returns_400():
    with TestClient(app) as client:
        resp = client.post("/messages/ingest", json={"sessionId": "s1"})
        assert resp.status_code == 400
        assert resp.json().get("detail") == "sessionId, messageId, and role are required"


def test_messages_ingest_one_off_enqueues_and_sets_group(monkeypatch: pytest.MonkeyPatch):
    # Force classifier to return one_off for this test
    def fake_cls(role: str, content: str, turn_count: int):
        return {"decision": "one_off", "confidence": 1.0}

    monkeypatch.setattr(main, "_classify_message_llm_stub", fake_cls)

    session_id = "sess_ingest_oneoff"
    with TestClient(app) as client:
        r = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m_any",
                "role": "user",
                "content": "any",
            },
        )
        assert r.status_code == 200
        j = r.json()
        assert j.get("state") == "idle"
        assert j.get("enqueued") is True
        assert isinstance(j.get("groupId"), str) and j["groupId"]
        gid = j["groupId"]

        payload = _poll_summary(client, session_id, gid)
        assert payload.get("latestGroupId") == gid
        summary = payload.get("summary", {})
        scores = summary.get("scores", {})
        job_err = summary.get("jobError")
        assert (isinstance(scores, dict) and len(scores) > 0) or job_err == "empty_scores_retries_exhausted"


def test_messages_ingest_low_confidence_fallback_applies_heuristics(monkeypatch: pytest.MonkeyPatch):
    # Force abstain/low confidence; heuristics should decide start when idle user message arrives
    def abstain_cls(role: str, content: str, turn_count: int):
        return {"decision": "abstain", "confidence": 0.2}

    monkeypatch.setattr(main, "_classify_message_llm_stub", abstain_cls)

    session_id = "sess_ingest_fallback"
    with TestClient(app) as client:
        r = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m0",
                "role": "user",
                "content": "Hello",
            },
        )
        assert r.status_code == 200
        j = r.json()
        # Heuristic for idle+user defaults to start
        assert j.get("state") == "active"
        assert j.get("turnCount") == 1
        assert j.get("enqueued") is False
