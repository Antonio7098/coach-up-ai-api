import asyncio
import json
import time
from typing import Dict, Any, List, Tuple

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import app.main as main


class _FakeHTTPResponse:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def read(self):
        return b"{}"


def _get_metric_count(name: str, labels: dict) -> float:
    val = REGISTRY.get_sample_value(name, labels)
    return float(val) if val is not None else 0.0


@pytest.mark.asyncio
async def test_per_skill_fanout_aggregation_and_metrics(monkeypatch: pytest.MonkeyPatch):
    # Enable provider path and configure two tracked skills
    monkeypatch.setenv("AI_ASSESS_ENABLED", "1")
    skills = [
        {"id": "s1", "name": "Clarity", "category": "communication"},
        {"id": "s2", "name": "Empathy", "category": "soft"},
    ]
    monkeypatch.setenv("ASSESS_TRACKED_SKILLS_JSON", json.dumps(skills))

    # Capture call timings to assert concurrency and provide per-skill outputs
    calls: List[Dict[str, Any]] = []

    class FakeAssessClient:
        provider_name = "test"
        model = "m"

        async def assess(self, transcript: List[Dict[str, Any]], rubric_version: str = "v2", request_id: str | None = None, skill: Dict[str, Any] | None = None) -> Dict[str, Any]:
            t0 = time.perf_counter()
            sid = (skill or {}).get("id")
            # Sleep different amounts per skill to allow overlap detection
            await asyncio.sleep(0.12 if sid == "s1" else 0.15)
            t1 = time.perf_counter()
            calls.append({"skill": sid, "start": t0, "end": t1})
            # Distinct scores per skill so aggregation is meaningful
            score = 0.2 if sid == "s1" else 0.8
            return {
                "rubricVersion": rubric_version,
                "categories": ["clarity"],
                "scores": {"clarity": score},
                "highlights": [f"h-{sid}"],
                "recommendations": [f"r-{sid}"],
                "meta": {"provider": "test", "skill": {k: (skill or {}).get(k) for k in ("id", "name", "category")}},
            }

    monkeypatch.setattr(main, "get_assess_client", lambda: FakeAssessClient())

    # Execute assessment job directly to get summary and scores
    scores, summary = await main._run_assessment_job("sess-fanout", "grp-1", request_id="rid-1", return_summary=True)

    # Concurrency: ensure calls overlap (start of one < end of other and vice versa)
    assert len(calls) == 2
    s1 = next(c for c in calls if c["skill"] == "s1")
    s2 = next(c for c in calls if c["skill"] == "s2")
    assert s1["start"] < s2["end"] and s2["start"] < s1["end"]

    # Aggregation: average of 0.2 and 0.8 is 0.5
    assert pytest.approx(scores.get("clarity"), rel=1e-3, abs=1e-6) == 0.5
    # Summary contains per-skill details
    sa = summary.get("skillAssessments") or []
    assert isinstance(sa, list) and len(sa) == 2
    assert summary.get("meta", {}).get("skillsCount") == 2


@pytest.mark.asyncio
async def test_per_skill_partial_failure_emits_error_metric(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_ASSESS_ENABLED", "1")
    skills = [
        {"id": "ok", "name": "OK", "category": "c"},
        {"id": "boom", "name": "Boom", "category": "c"},
    ]
    monkeypatch.setenv("ASSESS_TRACKED_SKILLS_JSON", json.dumps(skills))

    class FakeAssessClient:
        provider_name = "test"
        model = "m"
        async def assess(self, transcript, rubric_version="v2", request_id=None, skill=None):
            if (skill or {}).get("id") == "boom":
                raise RuntimeError("fail")
            return {
                "rubricVersion": rubric_version,
                "categories": ["clarity"],
                "scores": {"clarity": 0.7},
                "highlights": ["h"],
                "recommendations": ["r"],
                "meta": {"provider": "test"},
            }

    before_errors = _get_metric_count(
        "coachup_assessment_skill_errors_total", {"provider": "test", "model": "m", "reason": "exception"}
    )

    monkeypatch.setattr(main, "get_assess_client", lambda: FakeAssessClient())

    scores, summary = await main._run_assessment_job("sess-err", "grp-1", request_id=None, return_summary=True)

    after_errors = _get_metric_count(
        "coachup_assessment_skill_errors_total", {"provider": "test", "model": "m", "reason": "exception"}
    )
    # One error observed
    assert after_errors >= before_errors + 1
    # We still have a score from the successful skill
    assert scores.get("clarity") == pytest.approx(0.7, abs=1e-6)
    # Only one skill result included
    sa = summary.get("skillAssessments") or []
    assert len(sa) == 1


def test_persistence_includes_skill_assessments(monkeypatch: pytest.MonkeyPatch):
    # Enable provider path and configure skills; stub client returns fixed outputs
    monkeypatch.setenv("AI_ASSESS_ENABLED", "1")
    skills = [
        {"id": "s1", "name": "Clarity", "category": "communication"},
        {"id": "s2", "name": "Empathy", "category": "soft"},
    ]
    monkeypatch.setenv("ASSESS_TRACKED_SKILLS_JSON", json.dumps(skills))

    class FakeAssessClient:
        provider_name = "test"
        model = "m"
        async def assess(self, transcript, rubric_version="v2", request_id=None, skill=None):
            return {
                "rubricVersion": rubric_version,
                "categories": ["clarity"],
                "scores": {"clarity": 0.6},
                "highlights": [f"h-{(skill or {}).get('id')}"],
                "recommendations": ["r"],
                "meta": {"provider": "test", "skill": {k: (skill or {}).get(k) for k in ("id", "name", "category")}},
            }

    monkeypatch.setattr(main, "get_assess_client", lambda: FakeAssessClient())

    captured: Dict[str, Any] = {}

    def fake_urlopen(req, timeout):
        captured["url"] = getattr(req, "full_url", None)
        captured["headers"] = dict(req.header_items())
        captured["data"] = req.data
        captured["method"] = req.get_method()
        return _FakeHTTPResponse()

    monkeypatch.setattr(main.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(main, "PERSIST_ASSESSMENTS_URL", "http://example.com/finalize")
    monkeypatch.setattr(main, "PERSIST_ASSESSMENTS_SECRET", "secret")

    # Use API flow to exercise worker persistence path
    with TestClient(main.app) as client:
        r = client.post("/assessments/run", json={"sessionId": "sess-persist"})
        assert r.status_code == 200
        group_id = r.json()["groupId"]
        # Poll until a summary exists (the worker is synchronous in-process)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            # After the job completes, persistence should have been called
            if captured.get("data"):
                break
            time.sleep(0.05)

    assert captured.get("method") == "POST"
    body = json.loads((captured.get("data") or b"{}").decode("utf-8"))
    assert body.get("sessionId") == "sess-persist"
    assert body.get("groupId") == group_id
    summary = body.get("summary") or {}
    # Crucially: ensure skillAssessments are present and non-empty
    sa = summary.get("skillAssessments") or []
    assert isinstance(sa, list) and len(sa) == 2


@pytest.mark.asyncio
async def test_per_skill_latency_histogram(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_ASSESS_ENABLED", "1")
    skills = [{"id": "s1", "name": "A", "category": "c"}]
    monkeypatch.setenv("ASSESS_TRACKED_SKILLS_JSON", json.dumps(skills))

    class FakeAssessClient:
        provider_name = "test"
        model = "m"
        async def assess(self, transcript, rubric_version="v2", request_id=None, skill=None):
            await asyncio.sleep(0.05)
            return {"rubricVersion": "v2", "categories": ["c"], "scores": {"c": 0.9}, "highlights": ["h"], "recommendations": ["r"], "meta": {"provider": "test"}}

    before = _get_metric_count("coachup_assessment_skill_seconds_count", {"provider": "test", "model": "m", "rubric_version": "v2"})
    monkeypatch.setattr(main, "get_assess_client", lambda: FakeAssessClient())

    await main._run_assessment_job("sess-metrics", "grp-1", request_id=None, return_summary=False)

    after = _get_metric_count("coachup_assessment_skill_seconds_count", {"provider": "test", "model": "m", "rubric_version": "v2"})
    assert after >= before + 1
