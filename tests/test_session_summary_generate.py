import uuid
import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_summarizer():
    """Mock the summarizer to return predictable responses"""
    with patch('app.main.get_summary_client') as mock_get_client:
        mock_client = MagicMock()
        # Make summarize return a coroutine that resolves to a string
        async def mock_summarize(*args, **kwargs):
            return "This is a test summary of the conversation."
        mock_client.summarize = MagicMock(side_effect=mock_summarize)
        mock_client.provider_name = "mock"
        mock_client.model = "test-model"
        mock_get_client.return_value = mock_client
        yield mock_client


def test_generate_then_get_mock_provider(monkeypatch: pytest.MonkeyPatch, client: TestClient, mock_summarizer):
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


def test_get_session_summary_missing_session_id(client: TestClient):
    """Test GET endpoint with missing sessionId parameter"""
    response = client.get("/api/v1/session-summary")
    assert response.status_code == 200  # Returns empty summary for missing sessionId
    data = response.json()
    assert data.get("version") == 0
    assert data.get("updatedAt") == 0
    assert data.get("summaryText") == ""


def test_get_session_summary_nonexistent_session(client: TestClient):
    """Test GET endpoint for session that doesn't exist in store"""
    sid = str(uuid.uuid4())
    response = client.get("/api/v1/session-summary", params={"sessionId": sid})
    assert response.status_code == 200
    data = response.json()
    assert data.get("version") == 0
    assert data.get("updatedAt") == 0
    assert data.get("summaryText") == ""


def test_post_session_summary_missing_session_id(client: TestClient):
    """Test POST endpoint with missing sessionId"""
    payload = {
        "messages": [{"role": "user", "content": "Test message"}],
    }
    response = client.post("/api/v1/session-summary/generate", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "sessionId is required" in data.get("error", "")


def test_post_session_summary_empty_session_id(client: TestClient):
    """Test POST endpoint with empty sessionId"""
    payload = {
        "sessionId": "",
        "messages": [{"role": "user", "content": "Test message"}],
    }
    response = client.post("/api/v1/session-summary/generate", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "sessionId is required" in data.get("error", "")


def test_post_session_summary_invalid_json(client: TestClient):
    """Test POST endpoint with invalid JSON"""
    response = client.post("/api/v1/session-summary/generate",
                          data="invalid json",
                          headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    data = response.json()
    # FastAPI returns validation errors in 'detail' field, not 'error'
    assert "detail" in data


def test_post_session_summary_token_budget_enforcement(monkeypatch: pytest.MonkeyPatch, client: TestClient, mock_summarizer):
    """Test that token budget is properly enforced with minimums"""
    monkeypatch.setenv("AI_PROVIDER_SUMMARY", "mock")
    monkeypatch.setenv("AI_SUMMARY_TOKEN_BUDGET_MIN", "800")

    sid = str(uuid.uuid4())
    payload = {
        "sessionId": sid,
        "tokenBudget": 500,  # Below minimum
        "messages": [{"role": "user", "content": "Test"}],
    }

    response = client.post("/api/v1/session-summary/generate", json=payload)
    assert response.status_code == 200

    # Verify the summarizer was called with the enforced minimum budget
    mock_summarizer.summarize.assert_called_once()
    call_args = mock_summarizer.summarize.call_args[1]
    assert call_args["token_budget"] == 800  # Should be enforced to minimum


def test_post_session_summary_empty_ai_response(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    """Test handling of empty response from AI summarizer"""
    monkeypatch.setenv("AI_PROVIDER_SUMMARY", "mock")

    with patch('app.main.get_summary_client') as mock_get_client:
        mock_client = MagicMock()
        # Make summarize return a coroutine that resolves to empty string
        async def mock_summarize_empty(*args, **kwargs):
            return ""
        mock_client.summarize = mock_summarize_empty
        mock_client.provider_name = "mock"
        mock_client.model = "test-model"
        mock_get_client.return_value = mock_client

        sid = str(uuid.uuid4())
        payload = {
            "sessionId": sid,
            "messages": [{"role": "user", "content": "Test"}],
        }

        response = client.post("/api/v1/session-summary/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "empty"
        assert "X-Summary-Empty" in response.headers
        assert response.headers["X-Summary-Empty"] == "1"


def test_post_session_summary_versioning(monkeypatch: pytest.MonkeyPatch, client: TestClient, mock_summarizer):
    """Test that versioning increments correctly for multiple generations"""
    monkeypatch.setenv("AI_PROVIDER_SUMMARY", "mock")
    sid = str(uuid.uuid4())

    # First generation
    payload = {
        "sessionId": sid,
        "messages": [{"role": "user", "content": "First message"}],
    }
    r1 = client.post("/api/v1/session-summary/generate", json=payload)
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["version"] == 1

    # Second generation
    payload["prevSummary"] = "Previous summary text"
    r2 = client.post("/api/v1/session-summary/generate", json=payload)
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["version"] == 2

    # Verify GET returns the latest version
    r_get = client.get("/api/v1/session-summary", params={"sessionId": sid})
    assert r_get.status_code == 200
    data_get = r_get.json()
    assert data_get["version"] == 2


def test_post_session_summary_recent_messages_fallback(monkeypatch: pytest.MonkeyPatch, client: TestClient, mock_summarizer):
    """Test that recentMessages parameter works as fallback for messages"""
    monkeypatch.setenv("AI_PROVIDER_SUMMARY", "mock")
    sid = str(uuid.uuid4())

    payload = {
        "sessionId": sid,
        "recentMessages": [  # Using recentMessages instead of messages
            {"role": "user", "content": "Test with recentMessages"},
        ],
    }

    response = client.post("/api/v1/session-summary/generate", json=payload)
    assert response.status_code == 200

    # Verify the summarizer was called with the recentMessages
    mock_summarizer.summarize.assert_called_once()
    call_args = mock_summarizer.summarize.call_args[0]
    assert len(call_args[1]) == 1  # messages array
    assert call_args[1][0]["content"] == "Test with recentMessages"


def test_get_session_summary_etag_header(client: TestClient):
    """Test that GET endpoint returns proper ETag header"""
    sid = str(uuid.uuid4())

    # First request should return ETag
    response = client.get("/api/v1/session-summary", params={"sessionId": sid})
    assert "ETag" in response.headers
    etag = response.headers["ETag"]
    assert f"{sid}:0" in etag  # Format: sessionId:version



