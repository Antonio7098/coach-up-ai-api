import asyncio
import app.main as main


def _run_job(session_id: str, group_id: str):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(main._run_assessment_job(session_id, group_id, request_id=None))
    finally:
        loop.close()


def test_golden_determinism_session_group():
    expected = {
        "correctness": 0.55,
        "clarity": 0.62,
        "conciseness": 0.70,
        "fluency": 0.77,
    }
    scores = _run_job("determinism-session", "determinism-group")
    assert scores == expected


def test_golden_alpha_beta():
    expected = {
        "correctness": 0.88,
        "clarity": 0.95,
        "conciseness": 0.03,
        "fluency": 0.10,
    }
    scores = _run_job("alpha", "beta")
    assert scores == expected


def test_golden_sess_ingest_1_group_1():
    expected = {
        "correctness": 0.32,
        "clarity": 0.39,
        "conciseness": 0.47,
        "fluency": 0.54,
    }
    scores = _run_job("sess_ingest_1", "group_1")
    assert scores == expected


def test_golden_spr_002():
    expected = {
        "correctness": 0.81,
        "clarity": 0.88,
        "conciseness": 0.96,
        "fluency": 0.03,
    }
    scores = _run_job("spr", "002")
    assert scores == expected
