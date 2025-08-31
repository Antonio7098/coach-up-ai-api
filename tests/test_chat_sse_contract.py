import os
from typing import List

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _lower_headers(resp) -> dict:
    return {k.lower(): v for k, v in (resp.headers or {}).items()}


def test_chat_stream_headers_and_done_stub(monkeypatch: pytest.MonkeyPatch):
    # Force stub path to avoid external provider and make test deterministic
    monkeypatch.delenv("AI_CHAT_ENABLED", raising=False)

    with TestClient(app) as client:
        with client.stream(
            "GET",
            "/chat/stream?prompt=ping",
            headers={"accept": "text/event-stream", "x-request-id": "rid-abc"},
        ) as resp:
            # Headers
            h = _lower_headers(resp)
            assert "text/event-stream" in (h.get("content-type") or "")
            assert "charset" in (h.get("content-type") or "").lower()
            assert "no-cache" in (h.get("cache-control") or "").lower()
            assert "no-transform" in (h.get("cache-control") or "").lower()
            assert (h.get("connection") or "").lower().find("keep-alive") >= 0
            assert (h.get("x-accel-buffering") or "").lower() == "no"
            assert h.get("x-request-id") == "rid-abc"

            # Body contains data chunks and DONE terminator
            lines: List[str] = []
            for line in resp.iter_lines():
                if line is None:
                    continue
                lines.append(line)
            assert any(l.startswith("data:") for l in lines)
            assert any(l.strip() == "data: [DONE]" for l in lines)


def test_chat_stream_heartbeat_comments(monkeypatch: pytest.MonkeyPatch):
    # Force stub path and very frequent heartbeats to observe quickly
    monkeypatch.delenv("AI_CHAT_ENABLED", raising=False)
    monkeypatch.setenv("AI_CHAT_SSE_HEARTBEAT_SECONDS", "0.01")

    with TestClient(app) as client:
        with client.stream("GET", "/chat/stream?prompt=ping") as resp:
            saw_comment = False
            # Scan a bounded number of lines to find a comment heartbeat
            for idx, line in enumerate(resp.iter_lines()):
                if line is None:
                    continue
                if line.startswith(":"):
                    saw_comment = True
                    break
                if idx > 100:
                    break
        assert saw_comment, "Expected to observe at least one SSE comment heartbeat line"


def test_chat_stream_ttft_timeout_is_comment_and_done(monkeypatch: pytest.MonkeyPatch):
    # Enable provider path but force very short TTFT timeout to trigger fallback
    monkeypatch.setenv("AI_CHAT_ENABLED", "1")
    monkeypatch.setenv("AI_PROVIDER", "mock")
    monkeypatch.setenv("AI_CHAT_TTFT_TIMEOUT_SECONDS", "0.01")

    with TestClient(app) as client:
        with client.stream("GET", "/chat/stream?prompt=ping") as resp:
            saw_comment_notice = False
            saw_done = False
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith(": ") and "provider timeout" in line:
                    saw_comment_notice = True
                if line.strip() == "data: [DONE]":
                    saw_done = True
            assert saw_comment_notice, "Expected provider timeout notice as SSE comment (non-audible)"
            assert saw_done, "Expected DONE terminator even after TTFT timeout fallback"


