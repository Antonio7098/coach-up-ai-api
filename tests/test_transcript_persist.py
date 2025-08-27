import asyncio
import pytest

import app.main as main


@pytest.mark.asyncio
async def test_persist_interaction_session_only_excludes_group_id(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    async def fake_mutation(path, args, timeout=None):
        captured["path"] = path
        captured["args"] = args
        return {}

    # Force persistence on and avoid network/convex lookups
    monkeypatch.setattr(main, "_convex_mutation", fake_mutation)
    async def always_true(_sid):
        return True
    monkeypatch.setattr(main, "_should_persist_transcripts", always_true)

    await main._persist_interaction_if_configured(
        session_id="sess-only",
        group_id=None,
        message_id="m1",
        role="user",
        content="hello world",
        ts_ms=12345,
    )

    # We should have invoked the mutation with required fields only
    assert captured.get("path") == "functions/interactions:appendInteraction"
    args = captured.get("args") or {}
    assert args.get("sessionId") == "sess-only"
    assert args.get("messageId") == "m1"
    assert args.get("role") == "user"
    assert args.get("text") == "hello world"
    assert isinstance(args.get("ts"), int) and args["ts"] == 12345
    # groupId must be omitted entirely when not provided
    assert "groupId" not in args
    # contentHash should be present
    ch = args.get("contentHash")
    assert isinstance(ch, str) and len(ch) == 64


@pytest.mark.asyncio
async def test_persist_interaction_includes_group_id_when_present(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    async def fake_mutation(path, args, timeout=None):
        captured["path"] = path
        captured["args"] = args
        return {}

    monkeypatch.setattr(main, "_convex_mutation", fake_mutation)
    async def always_true2(_sid):
        return True
    monkeypatch.setattr(main, "_should_persist_transcripts", always_true2)

    await main._persist_interaction_if_configured(
        session_id="sess-1",
        group_id="grp-1",
        message_id="m2",
        role="assistant",
        content="ok",
        ts_ms=999,
    )

    args = captured.get("args") or {}
    assert args.get("groupId") == "grp-1"
    assert args.get("role") == "assistant"
