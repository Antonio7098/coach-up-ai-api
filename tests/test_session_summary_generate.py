import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_generate_then_get_mock_provider(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    # Force mock summarizer for determinism (no external calls)
    monkeypatch.setenv("AI_PROVIDER_SUMMARY", "mock")
    sid = str(uuid.uuid4())

    # POST generate
    payload = {
        "sessionId": sid,
        "prevSummary": "",
        "messages": [
            {"role": "user", "content": "Summarize: user set a goal to improve clarity."},
            {"role": "assistant", "content": "Coach suggested enunciation and pacing."},
        ],
        "tokenBudget": 600,
    }
    r1 = client.post("/api/v1/session-summary/generate", json=payload)
    assert r1.status_code == 200, r1.text
    data1 = r1.json()
    assert data1.get("sessionId") == sid
    assert isinstance(data1.get("version"), int)
    assert isinstance(data1.get("updatedAt"), int)
    assert isinstance(data1.get("text", ""), str)
    assert len(data1.get("text", "")) > 0

    # GET latest
    r2 = client.get(f"/api/v1/session-summary", params={"sessionId": sid})
    assert r2.status_code == 200, r2.text
    data2 = r2.json()
    assert data2.get("version") == data1.get("version")
    assert data2.get("summaryText", "") != ""
    # sanity: GET text should reflect a rolling summary string
    assert "Summary" in data2.get("summaryText", "") or len(data2.get("summaryText", "")) > 0



