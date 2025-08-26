import time
import asyncio
from typing import Dict

from fastapi.testclient import TestClient

from app.main import app, RUBRIC_V1_CATEGORIES


def test_assessments_run_and_get_flow():
    session_id = "test-session-123"
    with TestClient(app) as client:
        # Start a job
        resp = client.post("/assessments/run", json={"sessionId": session_id})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "accepted"
        group_id = data["groupId"]
        assert isinstance(group_id, str) and len(group_id) > 0

        # Poll for result
        deadline = time.time() + 5.0
        payload: Dict = {}
        while time.time() < deadline:
            r = client.get(f"/assessments/{session_id}")
            assert r.status_code == 200
            payload = r.json()
            if payload.get("latestGroupId") == group_id and payload.get("summary", {}).get("scores"):
                break
            time.sleep(0.05)

        assert payload.get("sessionId") == session_id
        assert payload.get("latestGroupId") == group_id
        summary = payload.get("summary")
        assert summary is not None
        assert summary.get("rubricVersion") == "v2"
        assert summary.get("categories") == RUBRIC_V1_CATEGORIES
        scores = summary.get("scores")
        assert isinstance(scores, dict)
        # Ensure all categories present and values within [0,1]
        for cat in RUBRIC_V1_CATEGORIES:
            assert cat in scores
            val = scores[cat]
            assert isinstance(val, (int, float))
            assert 0.0 <= val <= 1.0


def test_assessments_run_requires_session_id():
    with TestClient(app) as client:
        resp = client.post("/assessments/run", json={})
        assert resp.status_code == 400
        body = resp.json()
        assert body.get("detail") == "sessionId required"


def test_rubric_deterministic_hashing():
    # Calling the internal job twice with the same (sessionId, groupId)
    # should yield the same set of scores.
    import app.main as main

    session_id = "determinism-session"
    group_id = "determinism-group"

    loop = asyncio.new_event_loop()
    try:
        scores1 = loop.run_until_complete(main._run_assessment_job(session_id, group_id, request_id=None))
        scores2 = loop.run_until_complete(main._run_assessment_job(session_id, group_id, request_id=None))
    finally:
        loop.close()

    assert isinstance(scores1, dict) and isinstance(scores2, dict)
    assert scores1 == scores2
    # Values should be within [0,1]
    for v in scores1.values():
        assert 0.0 <= v <= 1.0


def test_metrics_endpoint_smoke():
    with TestClient(app) as client:
        r = client.get("/metrics")
        # When prometheus-client is installed, returns 200 with text/plain exposition format
        # Otherwise, returns 503 with a short text message
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            assert r.headers.get("content-type", "").startswith("text/plain")
            body = r.text
            # Spot-check at least one known metric name
            assert "coachup_assessment_job_seconds" in body
        else:
            assert "metrics unavailable" in r.text


def test_summary_meta_slice_and_highlights_from_transcript_slice():
    session_id = "slice-highlights-session"
    user_msg_1 = "I need a plan for two weeks."
    asst_msg_1 = "Here is a draft outline."
    asst_msg_end = "Good luck!"

    with TestClient(app) as client:
        # 1) User starts an interaction (LLM stub will suggest 'start' with low>low-threshold)
        r1 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m1",
                "role": "user",
                "content": user_msg_1,
            },
        )
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["state"] in ("active", "idle")  # should become active

        # 2) Assistant continues (heuristic 'continue')
        r2 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m2",
                "role": "assistant",
                "content": asst_msg_1,
            },
        )
        assert r2.status_code == 200

        # 3) Assistant ends with closing cue ('Good luck!') -> triggers assessment
        r3 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "m3",
                "role": "assistant",
                "content": asst_msg_end,
            },
        )
        assert r3.status_code == 200

        # Poll GET until summary available (should be immediate via sync path, but poll for safety)
        deadline = time.time() + 3.0
        payload = None
        while time.time() < deadline:
            gr = client.get(f"/assessments/{session_id}")
            assert gr.status_code == 200
            payload = gr.json()
            summary = (payload or {}).get("summary", {})
            scores = summary.get("scores")
            if scores:
                break
            time.sleep(0.05)

        assert payload is not None
        summary = payload["summary"]
        meta = summary.get("meta") or {}
        sl = (meta.get("slice") or {})

        # Validate meta.slice and messageCount reflect the 3-message window
        assert isinstance(sl.get("start_index"), int)
        assert isinstance(sl.get("end_index"), int)
        assert sl["start_index"] == 0
        assert sl["end_index"] == 3
        assert meta.get("messageCount") == 3

        # Highlights should include our user message and at least one assistant message
        highlights = summary.get("highlights") or []
        assert isinstance(highlights, list) and len(highlights) >= 2
        assert user_msg_1 in highlights
        # Assistant messages appear after user highlights
        assert any(h in highlights for h in [asst_msg_1, asst_msg_end])

        # Categories and scores present
        assert summary.get("categories") == RUBRIC_V1_CATEGORIES
        scores = summary.get("scores")
        for cat in RUBRIC_V1_CATEGORIES:
            assert cat in scores
            assert 0.0 <= scores[cat] <= 1.0


def test_summary_duration_ms_from_timestamps():
    session_id = "duration-session"
    # Explicit timestamps in ms
    t1 = 1_000
    t2 = 2_500
    t3 = 6_000

    with TestClient(app) as client:
        # User starts
        r1 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "d1",
                "role": "user",
                "content": "Plan please",
                "ts": t1,
            },
        )
        assert r1.status_code == 200

        # Assistant continues
        r2 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "d2",
                "role": "assistant",
                "content": "Outline...",
                "ts": t2,
            },
        )
        assert r2.status_code == 200

        # Assistant ends with closing cue
        r3 = client.post(
            "/messages/ingest",
            json={
                "sessionId": session_id,
                "messageId": "d3",
                "role": "assistant",
                "content": "Good luck!",
                "ts": t3,
            },
        )
        assert r3.status_code == 200

        # Fetch summary
        gr = client.get(f"/assessments/{session_id}")
        assert gr.status_code == 200
        payload = gr.json()
        summary = payload.get("summary", {})
        assert summary and summary.get("scores")
        meta = summary.get("meta") or {}
        assert meta.get("durationMs") == (t3 - t1)
