import time

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.main import app


def _get_metric_count(name: str, labels: dict) -> float:
    val = REGISTRY.get_sample_value(name, labels)
    return float(val) if val is not None else 0.0


def test_http_metrics_increment_on_2xx_and_4xx():
    client = TestClient(app)

    # Baselines
    before_ok = _get_metric_count(
        "coachup_http_requests_total", {"method": "GET", "path": "/health", "status_class": "2xx"}
    )
    before_dur_ok = _get_metric_count(
        "coachup_http_request_duration_seconds_count", {"method": "GET", "path": "/health"}
    )

    before_404 = _get_metric_count(
        "coachup_http_requests_total", {"method": "GET", "path": "/nope", "status_class": "4xx"}
    )
    before_dur_404 = _get_metric_count(
        "coachup_http_request_duration_seconds_count", {"method": "GET", "path": "/nope"}
    )

    # Hit a 2xx route
    r1 = client.get("/health")
    assert r1.status_code == 200

    # Hit a 4xx route
    r2 = client.get("/nope")
    assert r2.status_code == 404

    after_ok = _get_metric_count(
        "coachup_http_requests_total", {"method": "GET", "path": "/health", "status_class": "2xx"}
    )
    after_dur_ok = _get_metric_count(
        "coachup_http_request_duration_seconds_count", {"method": "GET", "path": "/health"}
    )

    after_404 = _get_metric_count(
        "coachup_http_requests_total", {"method": "GET", "path": "/nope", "status_class": "4xx"}
    )
    after_dur_404 = _get_metric_count(
        "coachup_http_request_duration_seconds_count", {"method": "GET", "path": "/nope"}
    )

    assert after_ok >= before_ok + 1
    assert after_dur_ok >= before_dur_ok + 1
    assert after_404 >= before_404 + 1
    assert after_dur_404 >= before_dur_404 + 1


def test_service_metrics_endpoint():
    client = TestClient(app)
    r = client.get("/service-metrics")
    assert r.status_code == 200
    body = r.json()
    assert set(["queueDepth", "workerConcurrency", "sessionCount", "resultsCount"]).issubset(body.keys())
