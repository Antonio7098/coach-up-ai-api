from fastapi import FastAPI, Request, Query, Body
from fastapi.openapi.utils import get_openapi
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse, Response
from starlette.middleware.cors import CORSMiddleware
from urllib.parse import parse_qs
from urllib.parse import unquote
import asyncio
import time
import json
import logging
import uuid
from typing import Optional, Dict, Any, Tuple, Set, List
import inspect
import urllib.request
import urllib.error
import os
import hashlib
from contextlib import asynccontextmanager
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    METRICS_ENABLED = True
except Exception:  # pragma: no cover - optional dependency fallback
    METRICS_ENABLED = False
    class _NoopMetric:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
    def Counter(*args, **kwargs):  # type: ignore
        return _NoopMetric()
    def Histogram(*args, **kwargs):  # type: ignore
        return _NoopMetric()
    def Gauge(*args, **kwargs):  # type: ignore
        return _NoopMetric()
    def generate_latest():  # type: ignore
        return b""
    CONTENT_TYPE_LATEST = "text/plain"

from app.middleware.request_id import RequestIdMiddleware
from app.providers.factory import get_chat_client, get_classifier_client


app = FastAPI(
    title="Coach Up AI API",
    description="AI service for assessments, multi-turn analysis, and provider integration.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
    allow_credentials=False,
)
app.add_middleware(RequestIdMiddleware)

logger = logging.getLogger("coach_up.ai.chat")

# Prometheus metrics
SQS_SEND_SECONDS = Histogram("coachup_sqs_send_seconds", "Duration of SQS send_message in seconds")
SQS_RECEIVE_SECONDS = Histogram("coachup_sqs_receive_seconds", "Duration of SQS receive_message in seconds")
SQS_DELETE_SECONDS = Histogram("coachup_sqs_delete_seconds", "Duration of SQS delete_message in seconds")
SQS_VISIBILITY_SECONDS = Histogram("coachup_sqs_change_visibility_seconds", "Duration of SQS change_message_visibility in seconds")
SQS_ENQUEUED_TOTAL = Counter("coachup_sqs_messages_enqueued_total", "SQS messages enqueued", ["status"])
SQS_POLLED_TOTAL = Counter("coachup_sqs_messages_polled_total", "SQS poll outcomes", ["outcome"])
SQS_DELETED_TOTAL = Counter("coachup_sqs_messages_deleted_total", "SQS message deletions", ["status"])
SQS_VISIBILITY_TOTAL = Counter("coachup_sqs_visibility_changes_total", "SQS visibility changes", ["status"])

ASSESS_JOB_SECONDS = Histogram("coachup_assessment_job_seconds", "Duration of assessment job in seconds")
ASSESS_ENQUEUE_LAT_SECONDS = Histogram("coachup_assessments_enqueue_latency_seconds", "Latency from enqueue to dequeue in seconds")
ASSESS_RETRIES_TOTAL = Counter("coachup_assessments_retries_total", "Assessment retries")
ASSESS_JOBS_TOTAL = Counter("coachup_assessments_jobs_total", "Assessment job outcomes", ["status"])
QUE_DEPTH = Gauge("coachup_assessments_queue_depth", "Assessments queue depth (in-memory provider only)", ["provider"])

# Rubric v1 categories (normalized [0,1])
# TODO(SPR-002 backend): Replace this stub with finalized rubric v1 definitions
# and ensure versioning is attached to persisted documents.
RUBRIC_V1_CATEGORIES = [
    "correctness",
    "clarity",
    "conciseness",
    "fluency",
]


# In-memory background worker & results store
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.assessment_queue = asyncio.Queue()  # type: ignore[attr-defined]
    app.state.assessment_results = {}  # type: ignore[attr-defined]
    # In-memory transcripts per session and group span indices for multi-turn slicing
    app.state.session_transcripts = {}  # type: ignore[attr-defined]
    # Keyed by (sessionId, groupId) -> {"start_index": int, "end_index": int}
    app.state.group_spans = {}  # type: ignore[attr-defined]
    # Initialize queue depth gauge for in-memory provider
    try:
        QUE_DEPTH.labels(provider="memory").set(app.state.assessment_queue.qsize())  # type: ignore[attr-defined]
    except Exception:
        pass
    # Spawn N workers based on WORKER_CONCURRENCY
    workers: List[asyncio.Task] = []
    for i in range(max(1, WORKER_CONCURRENCY)):
        # If SQS is enabled, use the SQS-backed worker; otherwise use in-memory queue worker
        if _sqs_enabled():
            workers.append(asyncio.create_task(_assessments_worker_sqs(app, i)))
        else:
            workers.append(asyncio.create_task(_assessments_worker(app, i)))
    app.state.assessment_worker_tasks = workers  # type: ignore[attr-defined]
    # Back-compat single task attribute
    if workers:
        app.state.assessment_worker_task = workers[0]  # type: ignore[attr-defined]
    # In-memory session interaction state and processed ids
    app.state.session_state: Dict[str, Dict[str, Any]] = {}
    # Track processed message ids per-session to avoid cross-session dedupe collisions
    app.state.processed_message_ids: Set[Tuple[str, str]] = set()
    # Track enqueue timestamps to measure queue latency per (sessionId, groupId)
    app.state.assessments_enqueued_ts: Dict[Tuple[str, str], float] = {}
    # Log worker configuration for observability
    try:
        logger.info(json.dumps({
            "event": "assessments_worker_config",
            "workerConcurrency": WORKER_CONCURRENCY,
        }))
    except Exception:
        pass

    try:
        yield
    finally:
        # Shutdown
        tasks: Optional[List[asyncio.Task]] = getattr(app.state, "assessment_worker_tasks", None)
        if tasks:
            for t in tasks:
                t.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass
        else:
            task = getattr(app.state, "assessment_worker_task", None)
            if task:
                task.cancel()
                try:
                    await task
                except Exception:
                    pass

# Register lifespan context to replace deprecated on_event hooks
app.router.lifespan_context = lifespan


async def _assessments_worker(app: FastAPI, worker_index: int = 0):
    """Background worker that processes assessment jobs from the queue.

    Job item: (session_id, group_id, request_id, tracked_hash)
    """
    queue: asyncio.Queue[Tuple[str, str, Optional[str], Optional[str]]] = app.state.assessment_queue  # type: ignore[attr-defined]
    results: Dict[str, Dict[str, Any]] = app.state.assessment_results  # type: ignore[attr-defined]
    while True:
        try:
            session_id, group_id, request_id, tracked_hash = await queue.get()
            # Compute enqueue latency if available
            enqueue_latency_ms: Optional[int] = None
            try:
                enq_map: Dict[Tuple[str, str], float] = getattr(app.state, "assessments_enqueued_ts", {})  # type: ignore[attr-defined]
                key = (session_id, group_id)
                t0 = enq_map.pop(key, None)
                if t0 is not None:
                    enqueue_latency_ms = int((time.perf_counter() - t0) * 1000)
            except Exception:
                enqueue_latency_ms = None
            # Metrics: observe enqueue latency and set in-memory queue depth gauge
            try:
                if enqueue_latency_ms is not None:
                    ASSESS_ENQUEUE_LAT_SECONDS.observe(enqueue_latency_ms / 1000.0)
                QUE_DEPTH.labels(provider="memory").set(queue.qsize())
            except Exception:
                pass
            # Log dequeue with current depth
            try:
                logger.info(json.dumps({
                    "event": "assessments_dequeue",
                    "requestId": request_id,
                    "trackedSkillIdHash": tracked_hash,
                    "sessionId": session_id,
                    "groupId": group_id,
                    "workerIndex": worker_index,
                    "queueDepth": queue.qsize(),
                    "enqueueLatencyMs": enqueue_latency_ms,
                }))
            except Exception:
                pass

            # Retry with exponential backoff if job fails (empty scores considered failure)
            attempt = 0
            scores: Dict[str, float] = {}
            summary_obj: Optional[Dict[str, Any]] = None
            while attempt < ASSESSMENTS_MAX_RETRIES:
                attempt += 1
                res = await _run_assessment_job(session_id, group_id, request_id, return_summary=True, tracked_hash=tracked_hash)
                if isinstance(res, tuple):
                    scores, summary_obj = res
                else:
                    scores = res
                    summary_obj = _compute_rubric_v1_summary_from_spans(session_id, group_id, scores, rubric_version="v1")
                if scores:
                    break
                backoff_ms = ASSESSMENTS_BACKOFF_BASE_MS * (2 ** (attempt - 1))
                try:
                    logger.info(json.dumps({
                        "event": "assessments_retry",
                        "requestId": request_id,
                        "trackedSkillIdHash": tracked_hash,
                        "sessionId": session_id,
                        "groupId": group_id,
                        "attempt": attempt,
                        "backoff_ms": backoff_ms,
                    }))
                except Exception:
                    pass
                try:
                    ASSESS_RETRIES_TOTAL.inc()
                except Exception:
                    pass
                await asyncio.sleep(backoff_ms / 1000.0)

            # Job outcome metric
            try:
                ASSESS_JOBS_TOTAL.labels(status="success" if bool(scores) else "failure").inc()
            except Exception:
                pass
            # Persist latest results in-memory (stub; replace with Convex in later sprint step)
            if summary_obj is None:
                summary_obj = _compute_rubric_v1_summary_from_spans(session_id, group_id, scores, rubric_version="v1")
            results[session_id] = {
                "latestGroupId": group_id,
                "summary": summary_obj,
            }
            # Optional HTTP persistence callback for integration with Convex/Next.js
            # Map to Convex finalize summary shape and include deterministic meta/categorical fields
            summary_for_persist = {
                "highlights": summary_obj.get("highlights", []) or ["placeholder"],
                "recommendations": summary_obj.get("recommendations", []) or ["placeholder"],
                "categories": summary_obj.get("categories", RUBRIC_V1_CATEGORIES),
                "scores": scores,
                "meta": summary_obj.get("meta", {}),
                "rubricVersion": summary_obj.get("rubricVersion", "v1"),
                "rubricKeyPoints": [
                    f"{cat}:{scores.get(cat)}" for cat in RUBRIC_V1_CATEGORIES
                ],
            }
            try:
                await _persist_assessment_if_configured(session_id, group_id, summary_for_persist, "v1", request_id)
            except Exception:
                pass
        except asyncio.CancelledError:
            break
        except Exception as e:
            try:
                logger.exception("assessments_worker_error: %s", e)
            except Exception:
                pass


@app.get("/health", tags=["meta"], description="Liveness endpoint for health checks.")
async def health():
    return {"status": "ok"}

@app.get("/metrics", tags=["meta"], include_in_schema=False)
async def metrics():
    if not METRICS_ENABLED:
        return Response(content="metrics unavailable", media_type="text/plain", status_code=503)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

async def _token_stream():
    # Longer stub stream to validate SSE pipeline and client-side TTS segmentation
    tokens = [
        "This ", "is ", "a ", "longer ", "streaming ", "sample ", "so ", "you ", "can ", "test ", "incremental ", "text-to-speech", ". ",
        "It ", "includes ", "multiple ", "sentences", ", ", "commas", ", ", "and ", "pauses", ". ",
        "As ", "each ", "sentence ", "arrives", ", ", "the ", "client ", "should ", "synthesize ", "and ", "enqueue ", "audio ", "chunks", ". ",
        "Try ", "interrupting ", "the ", "playback ", "to ", "confirm ", "barge-in ", "behavior", ". ",
        "Finally", ", ", "notice ", "how ", "punctuation ", "causes ", "natural ", "breaks ", "in ", "speech", ".\n",
    ]
    for token in tokens:
        yield f"data: {token}\n\n"
        await asyncio.sleep(0.15)
    yield "data: [DONE]\n\n"

@app.get("/chat/stream", tags=["chat"], description="SSE stream of chat tokens.")
async def chat_stream(
    request: Request,
    prompt: Optional[str] = Query(None, description="Optional user prompt for logging/testing"),
):
    start = time.perf_counter()
    first_token_s: Optional[float] = None
    provider_name: Optional[str] = None
    model_name: Optional[str] = None

    # Try to obtain request id from middleware or header
    request_id = getattr(getattr(request, "state", object()), "request_id", None) or request.headers.get("x-request-id")
    # Also obtain hashed tracked skill id for observability
    tracked_hash = getattr(getattr(request, "state", object()), "tracked_skill_id_hash", None)
    if not tracked_hash:
        try:
            _tsid = request.headers.get("x-tracked-skill-id")
            tracked_hash = hashlib.sha256(_tsid.encode("utf-8")).hexdigest() if _tsid else None
        except Exception:
            tracked_hash = None

    # Provider selection (env-gated)
    _enabled = os.getenv("AI_CHAT_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
    _model = os.getenv("AI_CHAT_MODEL", "").strip() or None

    async def instrumented_stream():
        nonlocal first_token_s
        nonlocal provider_name
        nonlocal model_name
        try:
            if _enabled:
                try:
                    client = get_chat_client(model=_model)
                    provider_name = getattr(client, "provider_name", None) or "unknown"
                    model_name = getattr(client, "model", None)
                    async for token in client.stream_chat(prompt or ""):
                        chunk = f"data: {token}\n\n"
                        if first_token_s is None:
                            first_token_s = time.perf_counter() - start
                            try:
                                logger.info(
                                    json.dumps(
                                        {
                                            "event": "chat_stream_first_token",
                                            "requestId": request_id,
                                            "trackedSkillIdHash": tracked_hash,
                                            "ttft_ms": int(first_token_s * 1000),
                                            "route": "/chat/stream",
                                            "prompt_present": bool(prompt),
                                            "provider": provider_name,
                                            "model": model_name,
                                        }
                                    )
                                )
                            except Exception:
                                pass
                        yield chunk
                    # End-of-stream marker
                    yield "data: [DONE]\n\n"
                    return
                except Exception:
                    # Fallback to stub stream on any provider error
                    provider_name = provider_name or "fallback_stub"
                    model_name = model_name or None
            async for chunk in _token_stream():
                if first_token_s is None:
                    first_token_s = time.perf_counter() - start
                    try:
                        logger.info(
                            json.dumps(
                                {
                                    "event": "chat_stream_first_token",
                                    "requestId": request_id,
                                    "trackedSkillIdHash": tracked_hash,
                                    "ttft_ms": int(first_token_s * 1000),
                                    "route": "/chat/stream",
                                    "prompt_present": bool(prompt),
                                    "provider": provider_name or ("provider" if _enabled else "stub"),
                                    "model": model_name,
                                }
                            )
                        )
                    except Exception:
                        pass
                yield chunk
        finally:
            total_s = time.perf_counter() - start
            try:
                logger.info(
                    json.dumps(
                        {
                            "event": "chat_stream_complete",
                            "requestId": request_id,
                            "trackedSkillIdHash": tracked_hash,
                            "ttft_ms": int(first_token_s * 1000) if first_token_s is not None else None,
                            "total_ms": int(total_s * 1000),
                            "route": "/chat/stream",
                            "prompt_present": bool(prompt),
                            "provider": provider_name or ("provider" if _enabled else "stub"),
                            "model": model_name,
                        }
                    )
                )
            except Exception:
                pass

    resp = StreamingResponse(instrumented_stream(), media_type="text/event-stream")
    if request_id:
        # Echo for clients/proxies that want to surface it
        resp.headers["X-Request-Id"] = str(request_id)
    return resp

def _compute_rubric_v1_summary_from_spans(
    session_id: str,
    group_id: str,
    scores: Dict[str, float],
    rubric_version: str = "v1",
) -> Dict[str, Any]:
    """Slice transcript by (sessionId, groupId) spans and build a deterministic summary.

    Returns a summary dict with highlights, recommendations, categories, scores, and meta.
    """
    try:
        transcripts: Dict[str, List[Dict[str, Any]]] = app.state.session_transcripts  # type: ignore[attr-defined]
        spans: Dict[Tuple[str, str], Dict[str, int]] = app.state.group_spans  # type: ignore[attr-defined]
    except Exception:
        transcripts = {}
        spans = {}
    events = transcripts.get(session_id, []) or []
    span = spans.get((session_id, group_id), {}) or {}
    start_idx = span.get("start_index", 0)
    end_idx = span.get("end_index", len(events))
    if start_idx < 0:
        start_idx = 0
    if end_idx is None or end_idx > len(events):
        end_idx = len(events)
    if end_idx < start_idx:
        end_idx = start_idx
    slice_events = events[start_idx:end_idx]

    user_msgs = [e.get("content", "") for e in slice_events if (e.get("role") == "user")]
    assistant_msgs = [e.get("content", "") for e in slice_events if (e.get("role") == "assistant")]
    message_count = len(slice_events)
    # Duration based on timestamps when present
    try:
        ts_vals = [int(e.get("ts")) for e in slice_events if isinstance(e.get("ts"), (int, float, str))]
        duration_ms = (max(ts_vals) - min(ts_vals)) if ts_vals else 0
    except Exception:
        duration_ms = 0

    def _first_non_empty(lst: List[str]) -> List[str]:
        out: List[str] = []
        for s in lst:
            s2 = (s or "").strip()
            if s2:
                out.append(s2)
            if len(out) >= 2:
                break
        return out

    highlights = (_first_non_empty(user_msgs) + _first_non_empty(assistant_msgs))[:3] or ["placeholder"]
    recommendations = [
        "Focus on clarity in follow-ups.",
        "Summarize key points before concluding.",
    ]

    return {
        "highlights": highlights,
        "recommendations": recommendations,
        "rubricVersion": rubric_version,
        "categories": RUBRIC_V1_CATEGORIES,
        "scores": scores,
        "meta": {
            "messageCount": message_count,
            "durationMs": duration_ms,
            "slice": {"start_index": start_idx, "end_index": end_idx},
        },
    }

async def _run_assessment_job(
    session_id: str,
    group_id: str,
    request_id: Optional[str],
    rubric_version: str = "v1",
    return_summary: bool = False,
    tracked_hash: Optional[str] = None,
):
    """Simulated multi-turn assessment job.

    Returns per-category scores in [0,1]. When `return_summary=True`, also returns
    a deterministic summary built via `_compute_rubric_v1_summary_from_spans()`.
    """
    # Rubric v1 pipeline (stubbed, deterministic):
    # - Slice buffered transcript by (sessionId, groupId) span indices
    # - Compute deterministic scores from slice content (hash-based)
    # - Produce a lightweight summary object from the same slice
    start = time.perf_counter()
    scores: Dict[str, float] = {}
    summary_obj: Optional[Dict[str, Any]] = None
    try:
        logger.info(
            json.dumps(
                {
                    "event": "assessments_job_start",
                    "requestId": request_id,
                    "trackedSkillIdHash": tracked_hash,
                    "sessionId": session_id,
                    "groupId": group_id,
                    "rubricVersion": rubric_version,
                }
            )
        )
        # Simulate rubric evaluation work
        await asyncio.sleep(0.05)

        # Retrieve transcript slice for this (session, group)
        try:
            transcripts: Dict[str, List[Dict[str, Any]]] = app.state.session_transcripts  # type: ignore[attr-defined]
            spans: Dict[Tuple[str, str], Dict[str, int]] = app.state.group_spans  # type: ignore[attr-defined]
        except Exception:
            transcripts = {}
            spans = {}
        events = transcripts.get(session_id, []) or []
        span = spans.get((session_id, group_id), {}) or {}
        start_idx = span.get("start_index", 0)
        end_idx = span.get("end_index", len(events))
        # Guard rails
        if start_idx < 0:
            start_idx = 0
        if end_idx is None or end_idx > len(events):
            end_idx = len(events)
        if end_idx < start_idx:
            end_idx = start_idx
        slice_events = events[start_idx:end_idx]

        # Deterministic seed from actual content when available; fallback to ids
        if slice_events:
            concat = "\n".join(f"{e.get('role','')}|{e.get('content','')}" for e in slice_events).encode("utf-8")
            seed_bytes = hashlib.sha256(concat).digest()
        else:
            seed_bytes = hashlib.sha256(f"{session_id}:{group_id}".encode("utf-8")).digest()
        h_base = int.from_bytes(seed_bytes[:4], "big") % 1000
        for i, cat in enumerate(RUBRIC_V1_CATEGORIES):
            scores[cat] = round(((h_base + (i * 73)) % 1000) / 1000.0, 2)

        # Optionally build deterministic summary from the same slice using the helper
        if return_summary:
            summary_obj = _compute_rubric_v1_summary_from_spans(
                session_id, group_id, scores, rubric_version=rubric_version
            )
        try:
            logger.info(
                json.dumps(
                    {
                        "event": "assessments_scores",
                        "requestId": request_id,
                        "trackedSkillIdHash": tracked_hash,
                        "sessionId": session_id,
                        "groupId": group_id,
                        "rubricVersion": rubric_version,
                        "scores": scores,
                        "span": {"start_index": start_idx, "end_index": end_idx},
                    }
                )
            )
        except Exception:
            pass
    except Exception:
        # If an unexpected error occurs, still log completion below
        pass
    total_s = time.perf_counter() - start
    try:
        logger.info(
            json.dumps(
                {
                    "event": "assessments_job_complete",
                    "requestId": request_id,
                    "trackedSkillIdHash": tracked_hash,
                    "sessionId": session_id,
                    "groupId": group_id,
                    "rubricVersion": rubric_version,
                    "total_ms": int(total_s * 1000),
                }
            )
        )
    except Exception:
        pass
    try:
        ASSESS_JOB_SECONDS.observe(total_s)
    except Exception:
        pass
    if return_summary:
        return scores, (summary_obj or {})
    return scores


# -----------------------------
# SQS queue integration (optional)
# -----------------------------

def _sqs_enabled() -> bool:
    try:
        return bool(USE_SQS and AWS_SQS_QUEUE_URL)
    except Exception:
        return False

def _get_sqs_client():
    if not _sqs_enabled():
        return None
    try:
        import boto3  # type: ignore
    except Exception:
        try:
            logger.warning(json.dumps({
                "event": "sqs_import_missing",
                "detail": "boto3 not installed; falling back to in-memory queue",
            }))
        except Exception:
            pass
        return None
    kwargs: Dict[str, Any] = {}
    if AWS_REGION:
        kwargs["region_name"] = AWS_REGION
    if AWS_ENDPOINT_URL_SQS:
        kwargs["endpoint_url"] = AWS_ENDPOINT_URL_SQS
    try:
        return boto3.client("sqs", **kwargs)
    except Exception as e:
        try:
            logger.warning(json.dumps({
                "event": "sqs_client_error",
                "error": str(e),
            }))
        except Exception:
            pass
        return None

async def _enqueue_assessment_job(app: FastAPI, session_id: str, group_id: str, request_id: Optional[str], tracked_hash: Optional[str] = None) -> None:
    # Record enqueue timestamp for latency measurement
    try:
        app.state.assessments_enqueued_ts[(session_id, group_id)] = time.perf_counter()  # type: ignore[attr-defined]
    except Exception:
        pass
    if _sqs_enabled():
        client = _get_sqs_client()
        if not client:
            # Fallback to in-memory if client not available
            await app.state.assessment_queue.put((session_id, group_id, request_id, tracked_hash))  # type: ignore[attr-defined]
            try:
                QUE_DEPTH.labels(provider="memory").set(app.state.assessment_queue.qsize())  # type: ignore[attr-defined]
            except Exception:
                pass
            return
        body_obj = {
            "sessionId": session_id,
            "groupId": group_id,
            "requestId": request_id,
        }
        if tracked_hash is not None:
            body_obj["trackedSkillIdHash"] = tracked_hash
        body = json.dumps(body_obj)
        dedup = f"{session_id}:{group_id}"
        try:
            loop = asyncio.get_running_loop()
            def _send():
                return client.send_message(
                    QueueUrl=AWS_SQS_QUEUE_URL,
                    MessageBody=body,
                    MessageGroupId=session_id,
                    MessageDeduplicationId=dedup,
                )
            t0 = time.perf_counter()
            await loop.run_in_executor(None, _send)
            try:
                SQS_SEND_SECONDS.observe(time.perf_counter() - t0)
                SQS_ENQUEUED_TOTAL.labels(status="ok").inc()
            except Exception:
                pass
            try:
                logger.info(json.dumps({
                    "event": "assessments_enqueue",
                    "requestId": request_id,
                    "trackedSkillIdHash": tracked_hash,
                    "sessionId": session_id,
                    "groupId": group_id,
                    "provider": "sqs",
                }))
            except Exception:
                pass
        except Exception as e:
            try:
                logger.warning(json.dumps({
                    "event": "assessments_enqueue_error",
                    "requestId": request_id,
                    "sessionId": session_id,
                    "groupId": group_id,
                    "provider": "sqs",
                    "error": str(e),
                }))
            except Exception:
                pass
            try:
                SQS_ENQUEUED_TOTAL.labels(status="error").inc()
            except Exception:
                pass
            # Fallback: best-effort in-memory enqueue
            await app.state.assessment_queue.put((session_id, group_id, request_id, tracked_hash))  # type: ignore[attr-defined]
            try:
                QUE_DEPTH.labels(provider="memory").set(app.state.assessment_queue.qsize())  # type: ignore[attr-defined]
            except Exception:
                pass
    else:
        await app.state.assessment_queue.put((session_id, group_id, request_id, tracked_hash))  # type: ignore[attr-defined]
        try:
            QUE_DEPTH.labels(provider="memory").set(app.state.assessment_queue.qsize())  # type: ignore[attr-defined]
        except Exception:
            pass

async def _assessments_worker_sqs(app: FastAPI, worker_index: int = 0):
    """Background worker that polls SQS for assessment jobs."""
    client = _get_sqs_client()
    if not client:
        # Fallback to in-memory worker loop
        await _assessments_worker(app, worker_index)
        return
    while True:
        try:
            loop = asyncio.get_running_loop()
            def _recv():
                return client.receive_message(
                    QueueUrl=AWS_SQS_QUEUE_URL,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20,
                    VisibilityTimeout=max(30, ASSESSMENTS_BACKOFF_BASE_MS // 1000 + 30),
                )
            t0 = time.perf_counter()
            resp = await loop.run_in_executor(None, _recv)
            try:
                SQS_RECEIVE_SECONDS.observe(time.perf_counter() - t0)
            except Exception:
                pass
            msgs = resp.get("Messages", []) or []
            if not msgs:
                try:
                    SQS_POLLED_TOTAL.labels(outcome="empty").inc()
                except Exception:
                    pass
                continue
            try:
                SQS_POLLED_TOTAL.labels(outcome="received").inc()
            except Exception:
                pass
            for m in msgs:
                receipt = m.get("ReceiptHandle")
                body = m.get("Body")
                session_id = group_id = request_id = None
                tracked_hash: Optional[str] = None
                try:
                    payload = json.loads(body or "{}")
                    session_id = payload.get("sessionId")
                    group_id = payload.get("groupId")
                    request_id = payload.get("requestId")
                    tracked_hash = payload.get("trackedSkillIdHash")
                except Exception:
                    pass
                if not session_id or not group_id:
                    # Malformed; delete to avoid poison
                    if receipt:
                        try:
                            def _del():
                                return client.delete_message(QueueUrl=AWS_SQS_QUEUE_URL, ReceiptHandle=receipt)
                            await loop.run_in_executor(None, _del)
                        except Exception:
                            pass
                    continue
                # Process job (reuse same logic as in-memory worker)
                enqueue_latency_ms: Optional[int] = None
                try:
                    t0 = getattr(app.state, "assessments_enqueued_ts", {}).pop((session_id, group_id), None)  # type: ignore[attr-defined]
                    if t0 is not None:
                        enqueue_latency_ms = int((time.perf_counter() - t0) * 1000)
                except Exception:
                    enqueue_latency_ms = None
                try:
                    if enqueue_latency_ms is not None:
                        ASSESS_ENQUEUE_LAT_SECONDS.observe(enqueue_latency_ms / 1000.0)
                except Exception:
                    pass
                try:
                    logger.info(json.dumps({
                        "event": "assessments_dequeue",
                        "requestId": request_id,
                        "trackedSkillIdHash": tracked_hash,
                        "sessionId": session_id,
                        "groupId": group_id,
                        "workerIndex": worker_index,
                        "provider": "sqs",
                        "enqueueLatencyMs": enqueue_latency_ms,
                    }))
                except Exception:
                    pass

                scores: Dict[str, float] = {}
                try:
                    res = await _run_assessment_job(session_id, group_id, request_id, return_summary=True, tracked_hash=tracked_hash)
                    if isinstance(res, tuple):
                        scores, summary_obj = res
                    else:
                        scores = res
                        summary_obj = _compute_rubric_v1_summary_from_spans(session_id, group_id, scores, rubric_version="v1")
                    # Persist results (same as in-memory worker)
                    app.state.assessment_results[session_id] = {  # type: ignore[attr-defined]
                        "latestGroupId": group_id,
                        "summary": summary_obj,
                    }
                    try:
                        ASSESS_JOBS_TOTAL.labels(status="success" if bool(scores) else "failure").inc()
                    except Exception:
                        pass
                    # Optional HTTP persistence callback
                    try:
                        summary_for_persist = {
                            "highlights": summary_obj.get("highlights", []) or ["placeholder"],
                            "recommendations": summary_obj.get("recommendations", []) or ["placeholder"],
                            "categories": summary_obj.get("categories", RUBRIC_V1_CATEGORIES),
                            "scores": scores,
                            "meta": summary_obj.get("meta", {}),
                            "rubricVersion": summary_obj.get("rubricVersion", "v1"),
                            "rubricKeyPoints": [
                                f"{cat}:{scores.get(cat)}" for cat in RUBRIC_V1_CATEGORIES
                            ],
                        }
                        await _persist_assessment_if_configured(session_id, group_id, summary_for_persist, "v1", request_id)
                    except Exception:
                        pass
                    # Delete message on success
                    if receipt:
                        try:
                            def _del_ok():
                                return client.delete_message(QueueUrl=AWS_SQS_QUEUE_URL, ReceiptHandle=receipt)
                            t1 = time.perf_counter()
                            await loop.run_in_executor(None, _del_ok)
                            try:
                                SQS_DELETE_SECONDS.observe(time.perf_counter() - t1)
                                SQS_DELETED_TOTAL.labels(status="ok").inc()
                            except Exception:
                                pass
                        except Exception:
                            try:
                                SQS_DELETED_TOTAL.labels(status="error").inc()
                            except Exception:
                                pass
                except Exception as e:
                    # Leave message for retry; optionally adjust visibility/backoff
                    try:
                        logger.exception("assessments_worker_sqs_error: %s", e)
                    except Exception:
                        pass
                    try:
                        ASSESS_JOBS_TOTAL.labels(status="failure").inc()
                    except Exception:
                        pass
                    try:
                        if receipt:
                            # Basic backoff
                            def _vis():
                                return client.change_message_visibility(
                                    QueueUrl=AWS_SQS_QUEUE_URL,
                                    ReceiptHandle=receipt,
                                    VisibilityTimeout=min(300, (ASSESSMENTS_BACKOFF_BASE_MS // 1000) + 30),
                                )
                            t2 = time.perf_counter()
                            await loop.run_in_executor(None, _vis)
                            try:
                                SQS_VISIBILITY_SECONDS.observe(time.perf_counter() - t2)
                                SQS_VISIBILITY_TOTAL.labels(status="ok").inc()
                            except Exception:
                                pass
                    except Exception:
                        try:
                            SQS_VISIBILITY_TOTAL.labels(status="error").inc()
                        except Exception:
                            pass
        except asyncio.CancelledError:
            break
        except Exception:
            # Swallow and continue
            await asyncio.sleep(1.0)

# -----------------------------
# Classifier (stub) and heuristics
# -----------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


CLASSIFIER_CONF_ACCEPT = _env_float("CLASSIFIER_CONF_ACCEPT", 0.7)
CLASSIFIER_CONF_LOW = _env_float("CLASSIFIER_CONF_LOW", 0.4)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    try:
        v = os.getenv(name, "1" if default else "0").strip().lower()
        return v in ("1", "true", "yes", "on")
    except Exception:
        return default

# Worker/runtime tuning
WORKER_CONCURRENCY = _env_int("WORKER_CONCURRENCY", 2)
ASSESSMENTS_MAX_RETRIES = _env_int("ASSESSMENTS_MAX_RETRIES", 3)
ASSESSMENTS_BACKOFF_BASE_MS = _env_int("ASSESSMENTS_BACKOFF_BASE_MS", 200)

# Queue provider (Phase 2) — SQS feature flag and configuration
USE_SQS = _env_bool("USE_SQS", False)
AWS_REGION = os.getenv("AWS_REGION", "").strip()
AWS_SQS_QUEUE_URL = os.getenv("AWS_SQS_QUEUE_URL", "").strip()
AWS_ENDPOINT_URL_SQS = os.getenv("AWS_ENDPOINT_URL_SQS", "").strip()

# Optional persistence callback (HTTP)
PERSIST_ASSESSMENTS_URL = os.getenv("PERSIST_ASSESSMENTS_URL", "").strip()
PERSIST_ASSESSMENTS_SECRET = os.getenv("PERSIST_ASSESSMENTS_SECRET", "").strip()

async def _persist_assessment_if_configured(
    session_id: str,
    group_id: str,
    summary: Dict[str, Any],
    rubric_version: str,
    request_id: Optional[str] = None,
) -> None:
    if not PERSIST_ASSESSMENTS_URL:
        return
    payload = {
        "sessionId": session_id,
        "groupId": group_id,
        "rubricVersion": rubric_version,
        "summary": summary,
    }
    headers = {
        "content-type": "application/json",
    }
    if request_id:
        headers["x-request-id"] = str(request_id)
    if PERSIST_ASSESSMENTS_SECRET:
        headers["authorization"] = f"Bearer {PERSIST_ASSESSMENTS_SECRET}"

    def _do_request():
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(PERSIST_ASSESSMENTS_URL, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                # Drain response to free the connection
                try:
                    resp.read()
                except Exception:
                    pass
        except Exception as e:
            try:
                logger.warning(json.dumps({
                    "event": "assessments_persist_error",
                    "requestId": request_id,
                    "sessionId": session_id,
                    "groupId": group_id,
                    "error": str(e),
                }))
            except Exception:
                pass

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _do_request)


async def _classify_message_llm_stub(role: str, content: str, turn_count: int) -> Dict[str, Any]:
    """Stub for LLM boundary classifier. Returns a decision with confidence.

    This should be replaced by a real LLM call. For now, emit low confidence
    or medium when clear cues are present.
    """
    text = (content or "").lower()
    if role == "assistant" and any(cue in text for cue in ["good luck", "let me know", "hope this helps", "anything else"]):
        return {"decision": "end", "confidence": 0.72, "reasons": "closing_cue"}
    if role == "user" and any(cue in text for cue in ["plan", "over the next", "for two weeks", "step-by-step"]):
        # Likely a new thread unless already active
        return {"decision": "start" if turn_count == 0 else "continue", "confidence": 0.65, "reasons": "planning_intent"}
    return {"decision": "abstain", "confidence": 0.2, "reasons": "unclear"}


def _heuristic_decision(role: str, content: str, active: bool) -> str:
    text = (content or "").lower()
    if role == "assistant":
        if any(cue in text for cue in ["good luck", "let me know", "anything else", "does that help", "glad to help"]):
            return "end"
        return "continue" if active else "ignore"
    # user
    if not active:
        return "start"
    return "continue"


def _ensure_group(session_state: Dict[str, Any]) -> None:
    if not session_state.get("groupId"):
        session_state["groupId"] = str(uuid.uuid4())
        session_state["turnCount"] = 0
        session_state["active"] = True


@app.post(
    "/messages/ingest",
    tags=["messages"],
    description="Ingest a chat message, classify boundary, update session state, and enqueue assessments when interactions end.",
)
async def messages_ingest(request: Request):
    request_id = getattr(getattr(request, "state", object()), "request_id", None) or request.headers.get("x-request-id")
    tracked_hash = getattr(getattr(request, "state", object()), "tracked_skill_id_hash", None)
    if not tracked_hash:
        try:
            _tsid = request.headers.get("x-tracked-skill-id")
            tracked_hash = hashlib.sha256(_tsid.encode("utf-8")).hexdigest() if _tsid else None
        except Exception:
            tracked_hash = None
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON body"}, status_code=400)

    session_id: Optional[str] = payload.get("sessionId")
    message_id: Optional[str] = payload.get("messageId")
    role: Optional[str] = payload.get("role")
    content: Optional[str] = payload.get("content")
    ts: Optional[int] = payload.get("ts")

    if not session_id or not message_id or not role:
        return JSONResponse({"detail": "sessionId, messageId, and role are required"}, status_code=400)

    # Idempotency: de-dupe by messageId
    processed: Set[Tuple[str, str]] = app.state.processed_message_ids  # type: ignore[attr-defined]
    key = (session_id, message_id)
    if key in processed:
        state = app.state.session_state.get(session_id, {"active": False, "groupId": None, "turnCount": 0})  # type: ignore[attr-defined]
        resp = {
            "state": "active" if state.get("active") else "idle",
            "groupId": state.get("groupId"),
            "turnCount": state.get("turnCount", 0),
            "enqueued": False,
            "deduped": True,
        }
        return JSONResponse(resp)

    # Session state init
    session_state = app.state.session_state.get(session_id)  # type: ignore[attr-defined]
    if not session_state:
        session_state = {"active": False, "groupId": None, "turnCount": 0, "lastTs": None}
        app.state.session_state[session_id] = session_state  # type: ignore[attr-defined]

    # Append to in-memory transcript buffer for this session (foundation for rubric v1)
    try:
        transcripts: Dict[str, List[Dict[str, Any]]] = app.state.session_transcripts  # type: ignore[attr-defined]
        lst = transcripts.setdefault(session_id, [])
        ts_ms: int
        try:
            ts_ms = int(ts) if ts is not None else int(time.time() * 1000)
        except Exception:
            ts_ms = int(time.time() * 1000)
        evt_idx = len(lst)
        lst.append({
            "messageId": message_id,
            "role": role,
            "content": content or "",
            "ts": ts_ms,
        })
    except Exception:
        # Do not fail ingestion if transcript buffering has issues
        evt_idx = -1  # sentinel

    # Classifier provider integration (env-gated) with robust fallback
    turn_count = int(session_state.get("turnCount", 0))
    cls_provider: Optional[str] = None
    cls_model: Optional[str] = None
    _cls_enabled = os.getenv("AI_CLASSIFIER_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
    try:
        # Back-compat toggle: DISABLE_CLASSIFIER=1 forces disabled
        if os.getenv("DISABLE_CLASSIFIER", "0").strip().lower() in ("1", "true", "yes", "on"):
            _cls_enabled = False
    except Exception:
        pass

    if _cls_enabled:
        try:
            _cls_model = os.getenv("AI_CLASSIFIER_MODEL", "").strip() or None
            client = get_classifier_client(model=_cls_model)
            cls_provider = getattr(client, "provider_name", None) or "unknown"
            cls_model = getattr(client, "model", None)
            cls = await client.classify(role, content or "", turn_count)
        except Exception:
            # Fallback to deterministic mock classifier
            try:
                client = get_classifier_client(provider="mock", model=os.getenv("AI_CLASSIFIER_MODEL", "").strip() or None)
                cls_provider = getattr(client, "provider_name", None) or "mock"
                cls_model = getattr(client, "model", None)
                cls = await client.classify(role, content or "", turn_count)
            except Exception:
                # Last resort: local stub
                _cls_res = _classify_message_llm_stub(role, content or "", turn_count)
                cls = await _cls_res if inspect.isawaitable(_cls_res) else _cls_res
    else:
        _cls_res = _classify_message_llm_stub(role, content or "", turn_count)
        cls = await _cls_res if inspect.isawaitable(_cls_res) else _cls_res

    decision: str = cls.get("decision", "abstain")
    confidence: float = float(cls.get("confidence", 0.0))
    accepted = confidence >= CLASSIFIER_CONF_ACCEPT
    low = CLASSIFIER_CONF_LOW <= confidence < CLASSIFIER_CONF_ACCEPT

    if not accepted:
        # Apply heuristic fallback or override if low or abstain
        heuristic = _heuristic_decision(role, content or "", bool(session_state.get("active")))
        decision = heuristic if confidence < CLASSIFIER_CONF_LOW else decision

    # Apply state machine
    enqueued = False
    if decision == "start":
        _ensure_group(session_state)
        session_state["turnCount"] = int(session_state.get("turnCount", 0)) + 1
        session_state["active"] = True
        # Mark start index for this interaction group span
        try:
            gid_now = session_state.get("groupId")
            if gid_now is not None and evt_idx >= 0:
                spans: Dict[Tuple[str, str], Dict[str, int]] = app.state.group_spans  # type: ignore[attr-defined]
                span = spans.setdefault((session_id, gid_now), {})
                if "start_index" not in span:
                    span["start_index"] = evt_idx
        except Exception:
            pass
        # Make latestGroupId immediately visible for polling clients
        try:
            gid_now = session_state.get("groupId")
            if gid_now:
                placeholder_summary = {
                    "highlights": ["placeholder"],
                    "recommendations": ["placeholder"],
                    "rubricVersion": "v1",
                    "categories": RUBRIC_V1_CATEGORIES,
                    "scores": {},
                }
                app.state.assessment_results[session_id] = {  # type: ignore[attr-defined]
                    "latestGroupId": gid_now,
                    "summary": placeholder_summary,
                }
        except Exception:
            pass
    elif decision == "continue":
        if not session_state.get("active"):
            _ensure_group(session_state)
        session_state["turnCount"] = int(session_state.get("turnCount", 0)) + 1
        session_state["active"] = True
        # Ensure a start index exists for this ongoing interaction
        try:
            gid_now = session_state.get("groupId")
            if gid_now is not None and evt_idx >= 0:
                spans = app.state.group_spans  # type: ignore[attr-defined]
                span = spans.setdefault((session_id, gid_now), {})
                if "start_index" not in span:
                    span["start_index"] = evt_idx
        except Exception:
            pass
        # Ensure polling clients see the latest groupId even mid-interaction
        try:
            gid_now = session_state.get("groupId")
            if gid_now:
                placeholder_summary = {
                    "highlights": ["placeholder"],
                    "recommendations": ["placeholder"],
                    "rubricVersion": "v1",
                    "categories": RUBRIC_V1_CATEGORIES,
                    "scores": {},
                }
                app.state.assessment_results[session_id] = {  # type: ignore[attr-defined]
                    "latestGroupId": gid_now,
                    "summary": placeholder_summary,
                }
        except Exception:
            pass
    elif decision == "end":
        if not session_state.get("active"):
            # nothing to end; ignore
            pass
        else:
            # Mark end index for the current group span
            try:
                gid_now_for_span = session_state.get("groupId")
                if gid_now_for_span is not None and evt_idx >= 0:
                    spans = app.state.group_spans  # type: ignore[attr-defined]
                    span = spans.setdefault((session_id, gid_now_for_span), {})
                    # End index is exclusive; use current transcript length if available
                    transcripts = app.state.session_transcripts  # type: ignore[attr-defined]
                    end_idx = len(transcripts.get(session_id, []))
                    span["end_index"] = end_idx
            except Exception:
                pass
            # enqueue assessment for this group
            try:
                gid_now = session_state.get("groupId")
                if gid_now:
                    # Record enqueue timestamp for latency measurement
                    try:
                        app.state.assessments_enqueued_ts[(session_id, gid_now)] = time.perf_counter()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                await _enqueue_assessment_job(app, session_id, gid_now, request_id, tracked_hash=tracked_hash)
                enqueued = True
                # Log enqueue event
                try:
                    logger.info(json.dumps({
                        "event": "assessments_enqueue",
                        "requestId": request_id,
                        "trackedSkillIdHash": tracked_hash,
                        "sessionId": session_id,
                        "groupId": gid_now,
                        "queueDepth": app.state.assessment_queue.qsize(),  # type: ignore[attr-defined]
                    }))
                except Exception:
                    pass
            except Exception:
                asyncio.create_task(_run_assessment_job(session_id, session_state.get("groupId"), request_id, tracked_hash=tracked_hash))
                enqueued = True
            # Also schedule a best-effort background job to ensure results are produced quickly
            if not _sqs_enabled():
                try:
                    asyncio.create_task(_run_assessment_job(session_id, session_state.get("groupId"), request_id, tracked_hash=tracked_hash))
                except Exception:
                    pass
            # And run once synchronously to ensure immediate availability
            if not _sqs_enabled():
                try:
                    gid_now = session_state.get("groupId")
                    if gid_now:
                        scores_sync, summary_obj = await _run_assessment_job(session_id, gid_now, request_id, return_summary=True, tracked_hash=tracked_hash)
                        # Persist in-memory results like worker does
                        try:
                            app.state.assessment_results[session_id] = {  # type: ignore[attr-defined]
                                "latestGroupId": gid_now,
                                "summary": summary_obj,
                            }
                        except Exception:
                            pass
                        # Optional HTTP persistence callback
                        try:
                            summary_for_persist = {
                                "highlights": summary_obj.get("highlights", []) or ["placeholder"],
                                "recommendations": summary_obj.get("recommendations", []) or ["placeholder"],
                                "categories": summary_obj.get("categories", RUBRIC_V1_CATEGORIES),
                                "scores": scores_sync,
                                "meta": summary_obj.get("meta", {}),
                                "rubricVersion": summary_obj.get("rubricVersion", "v1"),
                                "rubricKeyPoints": [
                                    f"{cat}:{scores_sync.get(cat)}" for cat in RUBRIC_V1_CATEGORIES
                                ],
                            }
                            await _persist_assessment_if_configured(session_id, gid_now, summary_for_persist, "v1", request_id)
                        except Exception:
                            pass
                except Exception:
                    pass
        # finalize interaction
        session_state["active"] = False
    elif decision == "one_off":
        # Enqueue single-turn assessment immediately
        gid = str(uuid.uuid4())
        # Record span as the single message for this one-off
        try:
            if evt_idx >= 0:
                spans = app.state.group_spans  # type: ignore[attr-defined]
                spans[(session_id, gid)] = {"start_index": evt_idx, "end_index": evt_idx + 1}
        except Exception:
            pass
        try:
            try:
                app.state.assessments_enqueued_ts[(session_id, gid)] = time.perf_counter()  # type: ignore[attr-defined]
            except Exception:
                pass
            await _enqueue_assessment_job(app, session_id, gid, request_id, tracked_hash=tracked_hash)
            enqueued = True
            try:
                logger.info(json.dumps({
                    "event": "assessments_enqueue",
                    "requestId": request_id,
                    "trackedSkillIdHash": tracked_hash,
                    "sessionId": session_id,
                    "groupId": gid,
                    "queueDepth": app.state.assessment_queue.qsize(),  # type: ignore[attr-defined]
                }))
            except Exception:
                pass
        except Exception:
            asyncio.create_task(_run_assessment_job(session_id, gid, request_id, tracked_hash=tracked_hash))
            enqueued = True
        # Also schedule a best-effort background job to ensure results are produced quickly
        if not _sqs_enabled():
            try:
                asyncio.create_task(_run_assessment_job(session_id, gid, request_id, tracked_hash=tracked_hash))
            except Exception:
                pass
        # And run once synchronously to ensure immediate availability
        if not _sqs_enabled():
            try:
                scores_sync, summary_obj = await _run_assessment_job(session_id, gid, request_id, return_summary=True, tracked_hash=tracked_hash)
                # Persist in-memory results like worker does
                try:
                    app.state.assessment_results[session_id] = {  # type: ignore[attr-defined]
                        "latestGroupId": gid,
                        "summary": summary_obj,
                    }
                except Exception:
                    pass
                # Optional HTTP persistence callback
                try:
                    summary_for_persist = {
                        "highlights": summary_obj.get("highlights", []) or ["placeholder"],
                        "recommendations": summary_obj.get("recommendations", []) or ["placeholder"],
                        "categories": summary_obj.get("categories", RUBRIC_V1_CATEGORIES),
                        "scores": scores_sync,
                        "meta": summary_obj.get("meta", {}),
                        "rubricVersion": summary_obj.get("rubricVersion", "v1"),
                        "rubricKeyPoints": [
                            f"{cat}:{scores_sync.get(cat)}" for cat in RUBRIC_V1_CATEGORIES
                        ],
                    }
                    await _persist_assessment_if_configured(session_id, gid, summary_for_persist, "v1", request_id)
                except Exception:
                    pass
            except Exception:
                pass
        session_state["active"] = False
        session_state["groupId"] = gid
        session_state["turnCount"] = 1
    else:
        # ignore/abstain: no state change
        pass

    # Update last timestamp and processed set
    session_state["lastTs"] = ts
    processed.add(key)

    # Build response
    resp = {
        "state": "active" if session_state.get("active") else "idle",
        "groupId": session_state.get("groupId"),
        "turnCount": session_state.get("turnCount", 0),
        "enqueued": enqueued,
    }
    try:
        logger.info(
            json.dumps(
                {
                    "event": "messages_ingest",
                    "requestId": request_id,
                    "trackedSkillIdHash": tracked_hash,
                    "sessionId": session_id,
                    "messageId": message_id,
                    "decision": decision,
                    "confidence": confidence,
                    "accepted": accepted,
                    "low": low,
                    "enqueued": enqueued,
                    "classifierEnabled": _cls_enabled,
                    "classifierProvider": cls_provider,
                    "classifierModel": cls_model,
                }
            )
        )
    except Exception:
        pass
    return JSONResponse(resp)

@app.post(
    "/assessments/run",
    tags=["assessments"],
    description="Start a multi-turn assessment job for a session; returns a groupId.",
)
async def assessments_run(request: Request):
    # TODO(SPR-002 backend): Enqueue real background worker (e.g. asyncio task queue)
    # - Accept optional groupHint for idempotency/coalescing
    # - Validate session ownership once auth is enabled
    # - Consider rate limits/idempotency keys
    # Accept sessionId from JSON body (embedded) or query string for robustness
    session_id: Optional[str] = None
    request_id = getattr(getattr(request, "state", object()), "request_id", None) or request.headers.get("x-request-id")
    tracked_hash = getattr(getattr(request, "state", object()), "tracked_skill_id_hash", None)
    if not tracked_hash:
        try:
            _tsid = request.headers.get("x-tracked-skill-id")
            tracked_hash = hashlib.sha256(_tsid.encode("utf-8")).hexdigest() if _tsid else None
        except Exception:
            tracked_hash = None
    # Read raw body once and parse permissively
    raw: bytes = b""
    try:
        raw = await request.body()
    except Exception:
        raw = b""
    if raw:
        # Try JSON first
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                session_id = parsed.get("sessionId")
        except Exception:
            # Try form-encoded
            try:
                form = parse_qs(raw.decode("utf-8"))
                vals = form.get("sessionId")
                if vals:
                    session_id = vals[0]
            except Exception:
                pass
    # debug print removed; using structured logger below
    try:
        logger.info(json.dumps({
            "event": "assessments_run_body_debug",
            "requestId": request_id,
            "trackedSkillIdHash": tracked_hash,
            "contentType": request.headers.get("content-type"),
            "rawLen": len(raw) if raw else 0,
            "rawPreview": raw.decode("utf-8", errors="ignore")[:200] if raw else "",
            "parsedSessionId": session_id,
        }))
    except Exception:
        pass
    if not session_id:
        session_id = request.query_params.get("sessionId")
    if not session_id:
        return JSONResponse({"detail": "sessionId required"}, status_code=400)

    group_id = str(uuid.uuid4())
    # Enqueue background job (worker consumes from queue)
    try:
        try:
            app.state.assessments_enqueued_ts[(session_id, group_id)] = time.perf_counter()  # type: ignore[attr-defined]
        except Exception:
            pass
        await _enqueue_assessment_job(app, session_id, group_id, request_id, tracked_hash=tracked_hash)
        try:
            logger.info(json.dumps({
                "event": "assessments_enqueue",
                "requestId": request_id,
                "trackedSkillIdHash": tracked_hash,
                "sessionId": session_id,
                "groupId": group_id,
                "queueDepth": app.state.assessment_queue.qsize(),  # type: ignore[attr-defined]
            }))
        except Exception:
            pass
    except Exception:
        # Fallback to immediate task if queue not ready (e.g., during early boot/tests)
        asyncio.create_task(_run_assessment_job(session_id, group_id, request_id, tracked_hash=tracked_hash))
    return {"groupId": group_id, "status": "accepted"}


@app.get(
    "/assessments/{sessionId}",
    tags=["assessments"],
    description="Fetch latest assessment summary for a session (stub).",
)
async def assessments_get(sessionId: str, request: Request):
    # Normalize ID in case clients send percent-encoded path segments
    try:
        decoded_id = unquote(sessionId)
    except Exception:
        decoded_id = sessionId
    # TODO(SPR-002 backend): Fetch from Convex instead of in-memory stub
    # - Query latest summary by sessionId
    # - Respect auth and user scoping
    results: Dict[str, Dict[str, Any]] = getattr(app.state, "assessment_results", {})  # type: ignore[attr-defined]
    # Try to obtain hashed tracked skill id for observability
    tracked_hash = getattr(getattr(request, "state", object()), "tracked_skill_id_hash", None)
    if not tracked_hash:
        try:
            _tsid = request.headers.get("x-tracked-skill-id")
            tracked_hash = hashlib.sha256(_tsid.encode("utf-8")).hexdigest() if _tsid else None
        except Exception:
            tracked_hash = None
    try:
        logger.info(json.dumps({
            "event": "assessments_get_lookup",
            "trackedSkillIdHash": tracked_hash,
            "sessionIdRaw": sessionId,
            "sessionIdDecoded": decoded_id,
            "results_has_raw": sessionId in results,
            "results_has_decoded": decoded_id in results,
        }))
    except Exception:
        pass
    if decoded_id in results or sessionId in results:
        data = results.get(decoded_id) or results.get(sessionId)
        return {"sessionId": sessionId, **data}
    # If not found, attempt to compute on-demand using last known groupId from session state
    try:
        ss_map: Dict[str, Dict[str, Any]] = getattr(app.state, "session_state", {})  # type: ignore[attr-defined]
        session_state: Dict[str, Any] = ss_map.get(decoded_id) or ss_map.get(sessionId)
        try:
            logger.info(json.dumps({
                "event": "assessments_get_state",
                "trackedSkillIdHash": tracked_hash,
                "has_state_decoded": bool(ss_map.get(decoded_id)),
                "has_state_raw": bool(ss_map.get(sessionId)),
                "groupId": (session_state or {}).get("groupId") if session_state else None,
            }))
        except Exception:
            pass
    except Exception:
        session_state = None  # type: ignore[assignment]
    if session_state and session_state.get("groupId"):
        gid = str(session_state.get("groupId"))
        # Compute scores synchronously and persist like the worker
        scores_sync, summary_obj = await _run_assessment_job(decoded_id, gid, request_id=None, return_summary=True, tracked_hash=tracked_hash)
        try:
            app.state.assessment_results[decoded_id] = {  # type: ignore[attr-defined]
                "latestGroupId": gid,
                "summary": summary_obj,
            }
        except Exception:
            pass
        return {"sessionId": sessionId, "latestGroupId": gid, "summary": summary_obj}
    # Fallback stub response (ensure shape and optionally echo last known groupId)
    try:
        ss_map2: Dict[str, Dict[str, Any]] = getattr(app.state, "session_state", {})  # type: ignore[attr-defined]
        session_state_fallback: Dict[str, Any] = ss_map2.get(decoded_id) or ss_map2.get(sessionId)
    except Exception:
        session_state_fallback = None  # type: ignore[assignment]
    return {
        "sessionId": sessionId,
        "latestGroupId": session_state_fallback.get("groupId") if session_state_fallback else None,
        "summary": {
            "highlights": ["placeholder"],
            "recommendations": ["placeholder"],
            "rubricVersion": "v1",
            "categories": RUBRIC_V1_CATEGORIES,
            "scores": {},
        },
    }


@app.get("/metrics", tags=["meta"], description="Lightweight service metrics for observability.")
async def metrics():
    try:
        queue: asyncio.Queue = app.state.assessment_queue  # type: ignore[attr-defined]
        workers = getattr(app.state, "assessment_worker_tasks", [])  # type: ignore[attr-defined]
        session_state: Dict[str, Any] = getattr(app.state, "session_state", {})  # type: ignore[attr-defined]
        results: Dict[str, Any] = getattr(app.state, "assessment_results", {})  # type: ignore[attr-defined]
        return {
            "queueDepth": queue.qsize(),
            "workerConcurrency": len(workers) or (1 if getattr(app.state, "assessment_worker_task", None) else 0),  # type: ignore[attr-defined]
            "sessionCount": len(session_state),
            "resultsCount": len(results),
        }
    except Exception:
        return {"queueDepth": None, "workerConcurrency": None, "sessionCount": None, "resultsCount": None}

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["tags"] = [
        {"name": "meta", "description": "Service metadata and liveness"},
        {"name": "chat", "description": "Realtime chat streaming (SSE)"},
        {"name": "assessments", "description": "Assessment runs and results"},
        {"name": "messages", "description": "Message ingestion and boundary classification"},
    ]
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local dev"}
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[assignment]
