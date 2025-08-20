import time
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
        assert summary.get("rubricVersion") == "v1"
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
