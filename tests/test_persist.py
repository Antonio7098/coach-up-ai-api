import asyncio
import json
from types import SimpleNamespace

import app.main as main


class _FakeResponse:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def read(self):
        return b"{}"


def test_persist_assessment_if_configured_noop_when_url_missing(monkeypatch):
    called = {"urlopen": False}

    def fake_urlopen(*args, **kwargs):
        called["urlopen"] = True
        raise AssertionError("urlopen should not be called when URL is empty")

    monkeypatch.setattr(main.urllib.request, "urlopen", fake_urlopen)
    # Ensure URL is empty
    monkeypatch.setattr(main, "PERSIST_ASSESSMENTS_URL", "")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            main._persist_assessment_if_configured(
                session_id="s1",
                group_id="g1",
                summary={"highlights": ["h"], "recommendations": ["r"], "rubricKeyPoints": ["k:0.5"]},
                rubric_version="v1",
                request_id="rid-1",
            )
        )
    finally:
        loop.close()

    assert called["urlopen"] is False


def test_persist_assessment_if_configured_sends_expected_payload_and_headers(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        # Capture request details
        captured["url"] = getattr(req, "full_url", None)
        captured["headers"] = dict(req.header_items())
        captured["data"] = req.data
        captured["method"] = req.get_method()
        return _FakeResponse()

    # Configure URL and secret
    monkeypatch.setattr(main.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(main, "PERSIST_ASSESSMENTS_URL", "http://example.com/finalize")
    monkeypatch.setattr(main, "PERSIST_ASSESSMENTS_SECRET", "topsecret")

    payload_summary = {
        "highlights": ["a"],
        "recommendations": ["b"],
        "rubricKeyPoints": ["correctness:0.9"],
    }

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            main._persist_assessment_if_configured(
                session_id="sess-123",
                group_id="grp-456",
                summary=payload_summary,
                rubric_version="v1",
                request_id="req-xyz",
            )
        )
    finally:
        loop.close()

    assert captured.get("url") == "http://example.com/finalize"
    assert captured.get("method") == "POST"

    headers = captured.get("headers") or {}
    # Normalize header keys to lowercase for comparison since urllib may title-case them
    headers = {k.lower(): v for k, v in headers.items()}
    assert headers.get("content-type") == "application/json"
    assert headers.get("authorization") == "Bearer topsecret"
    assert headers.get("x-request-id") == "req-xyz"

    body = json.loads((captured.get("data") or b"{}").decode("utf-8"))
    assert body == {
        "sessionId": "sess-123",
        "groupId": "grp-456",
        "rubricVersion": "v1",
        "summary": payload_summary,
    }
