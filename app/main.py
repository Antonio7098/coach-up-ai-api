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
import base64
from contextlib import asynccontextmanager
import io
import re
import wave
import httpx

# Load environment variables from .env if available, but avoid during pytest to keep tests deterministic
try:
    from dotenv import load_dotenv
    if "PYTEST_CURRENT_TEST" not in os.environ:
        load_dotenv()
except Exception:
    pass

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
from app.providers.factory import get_chat_client, get_classifier_client, get_assess_client, get_summary_client


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

# HTTP metrics middleware
@app.middleware("http")
async def _http_metrics_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    method = request.method
    path = request.url.path
    status_code = 500
    try:
        response = await call_next(request)
        status_code = getattr(response, "status_code", 500)
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        try:
            status_class = f"{status_code // 100}xx"
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status_class=status_class).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(time.perf_counter() - t0)
        except Exception:
            pass

# HTTP request metrics
HTTP_REQUESTS_TOTAL = Counter(
    "coachup_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_class"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "coachup_http_request_duration_seconds",
    "Duration of HTTP requests in seconds",
    ["method", "path"],
)

logger = logging.getLogger("coach_up.ai.chat")
# Ensure our application logger emits under Uvicorn:
# - honor LOG_LEVEL env (default INFO)
# - attach a StreamHandler if none present
# - disable propagate to avoid duplicate logs with Uvicorn root handlers
try:
    _lvl_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    _lvl = getattr(logging, _lvl_name, logging.INFO)
except Exception:
    _lvl = logging.INFO
logger.setLevel(_lvl)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(_lvl)
    _h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_h)
# Avoid double logging if root/uvicorn also handle propagation
logger.propagate = False

def _sse_data_event(text: str) -> str:
    """
    Encode text as a well-formed SSE data event.
    Splits on newlines and prefixes each with 'data: ', ending with a blank line.
    Prevents malformed events when tokens contain newlines.
    """
    try:
        lines = str(text).splitlines()
    except Exception:
        lines = [str(text)]
    if not lines:
        return "data: \n\n"
    return "".join(f"data: {line}\n" for line in lines) + "\n"

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
# Count cases where a provider run yields empty scores (used for alerting/dashboards)
ASSESS_EMPTY_SCORES_TOTAL = Counter(
    "coachup_assessments_empty_scores_total",
    "Assessment runs that produced empty scores",
    ["provider", "model", "rubric_version", "reason"],
)
QUE_DEPTH = Gauge("coachup_assessments_queue_depth", "Assessments queue depth (in-memory provider only)", ["provider"])

# Aggregated tokens and cost metrics (by provider/model/rubric)
ASSESS_TOKENS_TOTAL = Counter(
    "coachup_assessment_tokens_total",
    "Total assessment tokens",
    ["direction", "provider", "model", "rubric_version"],
)
ASSESS_COST_USD_TOTAL = Counter(
    "coachup_assessment_cost_usd_total",
    "Total assessment cost in USD",
    ["provider", "model", "rubric_version"],
)

# Per-skill assessment metrics
ASSESS_SKILL_SECONDS = Histogram(
    "coachup_assessment_skill_seconds",
    "Duration of per-skill assessment in seconds",
    ["provider", "model", "rubric_version"],
)
ASSESS_SKILL_ERRORS_TOTAL = Counter(
    "coachup_assessment_skill_errors_total",
    "Per-skill provider assessment errors",
    ["provider", "model", "reason"],
)

# Chat streaming metrics
CHAT_TTFT_SECONDS = Histogram(
    "coachup_chat_ttft_seconds",
    "Time to first token for chat streaming in seconds",
    ["provider", "model"],
)
CHAT_TOTAL_SECONDS = Histogram(
    "coachup_chat_total_seconds",
    "Total chat streaming duration in seconds",
    ["provider", "model"],
)

# Session summary metrics (independent of assessments)
SUMMARY_JOBS_TOTAL = Counter(
    "coachup_summary_jobs_total",
    "Session summary job outcomes",
    ["status"],
)
SUMMARY_JOB_SECONDS = Histogram(
    "coachup_summary_job_seconds",
    "Duration of session summary jobs in seconds",
)
SUMMARY_ENQUEUE_LAT_SECONDS = Histogram(
    "coachup_summary_enqueue_latency_seconds",
    "Latency from enqueue to dequeue for session summary in seconds",
)
SUMMARY_QUEUE_DEPTH = Gauge(
    "coachup_summary_queue_depth",
    "Session summary queue depth (in-memory)",
)

# Transcript persistence & classifier context metrics
TRANSCRIPT_PERSIST_TOTAL = Counter(
    "coachup_transcript_persist_total",
    "Transcript persistence outcomes",
    ["outcome", "role"],
)
TRANSCRIPT_PERSIST_SECONDS = Histogram(
    "coachup_transcript_persist_seconds",
    "Duration of transcript persistence to Convex in seconds",
)
CLASSIFIER_CONTEXT_BUILD_TOTAL = Counter(
    "coachup_classifier_context_build_total",
    "Classifier context build outcomes",
    ["outcome"],
)
CLASSIFIER_CONTEXT_MESSAGES = Histogram(
    "coachup_classifier_context_messages",
    "Number of messages included in classifier context",
)
CLASSIFIER_CONTEXT_LENGTH_CHARS = Histogram(
    "coachup_classifier_context_length_chars",
    "Length of classifier context string in characters",
)

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
    # In-memory TTS audio store for benchmarking stub
    app.state.tts_audio = {}  # type: ignore[attr-defined]
    # Session summary: queue and latest summaries per session
    app.state.summary_queue = asyncio.Queue()  # type: ignore[attr-defined]
    app.state.session_summaries = {}  # type: ignore[attr-defined]
    app.state.summary_enqueued_ts = {}  # type: ignore[attr-defined]
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
    # Spawn session summary workers (independent of assessments)
    summary_workers: List[asyncio.Task] = []
    if SUMMARY_ENABLED:
        for i in range(max(1, SUMMARY_WORKER_CONCURRENCY)):
            summary_workers.append(asyncio.create_task(_summary_worker(app, i)))
    app.state.summary_worker_tasks = summary_workers  # type: ignore[attr-defined]
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
        # Shutdown summary workers
        try:
            s_tasks: Optional[List[asyncio.Task]] = getattr(app.state, "summary_worker_tasks", None)
            if s_tasks:
                for t in s_tasks:
                    t.cancel()
                try:
                    await asyncio.gather(*s_tasks, return_exceptions=True)
                except Exception:
                    pass
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
                    summary_obj = _compute_rubric_v1_summary_from_spans(session_id, group_id, scores, rubric_version="v2")
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
            # If retries exhausted and still empty, attach a jobError field for observability — even if provider returned a summary
            if not scores:
                base = summary_obj if isinstance(summary_obj, dict) else {}
                categories = base.get("categories") or RUBRIC_V1_CATEGORIES
                rubric_version_final = base.get("rubricVersion") or "v2"
                scores_base = base.get("scores") or {}
                highlights = base.get("highlights") or ["assessment unavailable"]
                recommendations = base.get("recommendations") or ["Try again later or adjust provider settings."]
                meta = base.get("meta") or {"skillsCount": 0}
                meta.setdefault("skillsCount", 0)
                summary_obj = {
                    **base,
                    "rubricVersion": rubric_version_final,
                    "categories": categories,
                    "scores": scores_base,
                    "highlights": highlights,
                    "recommendations": recommendations,
                    "jobError": "empty_scores_retries_exhausted",
                    "meta": meta,
                }
            elif summary_obj is None:
                summary_obj = _compute_rubric_v1_summary_from_spans(session_id, group_id, scores, rubric_version="v2")
            results[session_id] = {
                "latestGroupId": group_id,
                "summary": summary_obj,
            }
            # Optional HTTP persistence callback for integration with Convex/Next.js
            try:
                v2_payload = _build_v2_persist_payload(summary_obj or {}, scores)
                await _persist_assessment_if_configured(session_id, group_id, v2_payload, "v2", request_id)
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
        yield _sse_data_event(token)
        await asyncio.sleep(0.15)
    yield "data: [DONE]\n\n"

@app.get("/chat/stream", tags=["chat"], description="SSE stream of chat tokens.")
async def chat_stream(
    request: Request,
    prompt: Optional[str] = Query(None, description="Optional user prompt for logging/testing"),
    session_id: Optional[str] = Query(None, description="Optional session id to enable multi-turn context"),
    history: Optional[str] = Query(None, description="Optional base64url-encoded JSON array of messages"),
    mock_user_data: Optional[str] = Query(None, description="Optional base64url-encoded mock user data for testing"),
):
    start = time.perf_counter()
    first_token_s: Optional[float] = None
    provider_name: Optional[str] = None
    model_name: Optional[str] = None

    # Try to obtain request id from middleware or header
    request_id = getattr(getattr(request, "state", object()), "request_id", None) or request.headers.get("x-request-id")

    # Debug: log incoming parameters
    logger.info(f"[DEBUG] chat_stream called with: prompt={bool(prompt)}, session_id={bool(session_id)}, history={bool(history)}, mock_user_data={bool(mock_user_data)}")
    # Also obtain hashed tracked skill id for observability
    tracked_hash = getattr(getattr(request, "state", object()), "tracked_skill_id_hash", None)
    if not tracked_hash:
        try:
            _tsid = request.headers.get("x-tracked-skill-id")
            tracked_hash = _skill_hash(_tsid) if _tsid else None
        except Exception:
            tracked_hash = None

    # Provider selection (env-gated)
    _enabled = os.getenv("AI_CHAT_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
    _model = os.getenv("AI_CHAT_MODEL", "").strip() or None
    
    # Debug: log provider configuration
    logger.info(f"[DEBUG] AI_CHAT_ENABLED={_enabled}, AI_CHAT_MODEL={_model}")

    # Pre-resolve provider/model so we can emit reliable headers on the response
    client_outer = None
    if _enabled:
        try:
            client_outer = get_chat_client(model=_model)
            provider_name = getattr(client_outer, "provider_name", None) or provider_name
            model_name = getattr(client_outer, "model", None) or model_name
        except Exception as e:
            # Log client construction error for better diagnostics instead of silently swallowing
            try:
                logger.exception(
                    json.dumps(
                        {
                            "event": "chat_stream_client_build_error",
                            "route": "/chat/stream",
                            "requestId": getattr(getattr(request, "state", object()), "request_id", None) or request.headers.get("x-request-id"),
                            "provider": provider_name or (os.getenv("AI_PROVIDER_CHAT") or os.getenv("AI_PROVIDER") or "unknown"),
                            "model": _model,
                            "error": str(e),
                        }
                    )
                )
            except Exception:
                pass
            client_outer = None

    async def instrumented_stream():
        nonlocal first_token_s
        nonlocal provider_name
        nonlocal model_name
        try:
            # Define stub stream upfront so provider fallback can reuse it safely
            async def _stub_stream():
                yield _sse_data_event("Hello! I'm in stub mode.")
                await asyncio.sleep(0.05)
                yield _sse_data_event("Chat functionality is disabled.")
                await asyncio.sleep(0.05)
                yield _sse_data_event("Set AI_CHAT_ENABLED=1 to enable real chat.")
                await asyncio.sleep(0.15)
                yield "data: [DONE]\n\n"

            _token_stream = None
            provider_setup_error: Optional[str] = None
            if _enabled:
                try:
                    client = client_outer or get_chat_client(model=_model)
                    provider_name = getattr(client, "provider_name", None) or "unknown"
                    model_name = getattr(client, "model", None)
                    # Build context from client-passed history when available; otherwise fallback to server transcripts and user profile/goals.
                    # Accept both "sessionId" and "session_id" query keys for flexibility
                    sid = session_id or request.query_params.get("session_id") or request.query_params.get("sessionId")
                    ctx = ""
                    ctx_source = "none"
                    hist_b64 = request.query_params.get("history")
                    # Try client-provided history (base64url JSON array of {role, content})
                    if hist_b64:
                        try:
                            s = str(hist_b64)
                            pad = "=" * (-len(s) % 4)
                            decoded = base64.urlsafe_b64decode((s + pad).encode("utf-8")).decode("utf-8", errors="ignore")
                            arr = json.loads(decoded)
                            if isinstance(arr, list) and arr:
                                # Limit number of messages for safety
                                try:
                                    limit = _chat_context_limit_default()
                                except Exception:
                                    limit = 10
                                items = arr[-max(1, min(200, limit)) :]
                                lines: List[str] = []
                                lines_turns: List[str] = []
                                lines_unknown: List[str] = []
                                # Env-configurable caps
                                _item_word_cap = _chat_item_word_cap_default()
                                _total_word_cap = _chat_total_word_cap_default()
                                _item_char_cap = _chat_item_char_cap_default()
                                _total_char_cap = _chat_total_char_cap_default()
                                for it in items:
                                    if not isinstance(it, dict):
                                        continue
                                    role = str((it.get("role") or "")).lower()
                                    tag = "u" if role == "user" else ("a" if role == "assistant" else "?")
                                    txt_raw = str((it.get("content") or "")).replace("\n", " ").strip()
                                    # Per-item truncation: prefer words, fallback to chars
                                    if _item_word_cap > 0:
                                        txt = _truncate_by_words(txt_raw, _item_word_cap)
                                    else:
                                        _cap = max(1, int(_item_char_cap or 240))
                                        txt = (txt_raw[:max(0, _cap - 3)] + "...") if len(txt_raw) > _cap else txt_raw
                                    if txt:
                                        line = f"{tag}: {txt}"
                                        lines.append(line)
                                        if tag == "?":
                                            lines_unknown.append(line)
                                        else:
                                            lines_turns.append(line)
                                # Present context in sections for readability
                                sections: List[str] = []
                                if lines_unknown:
                                    # Strip leading "?: " for display
                                    unknown_fmt = ["- " + l[3:] if l.startswith("?: ") else ("- " + l) for l in lines_unknown]
                                    sections.append("Session summary:\n" + "\n".join(unknown_fmt))
                                if lines_turns:
                                    turns_fmt = ["- " + l for l in lines_turns]
                                    sections.append("Recent turns:\n" + "\n".join(turns_fmt))
                                ctx = "\n\n".join(sections) if sections else "\n".join(["- " + l for l in lines])
                                # Total truncation: prefer words, fallback to chars
                                if _total_word_cap > 0:
                                    ctx = _truncate_by_words(ctx, _total_word_cap)
                                else:
                                    _tcap = max(1, int(_total_char_cap or 6000))
                                    if len(ctx) > _tcap:
                                        ctx = ctx[:max(0, _tcap - 3)] + "..."
                                if ctx:
                                    ctx_source = "client"
                        except Exception:
                            pass
                    # Fallback to server-side transcript summary when client history not present/invalid (with timeout)
                    user_profile_text = ""
                    goals_text = ""
                    if not ctx and sid:
                        try:
                            _timeout_s = float(os.getenv("AI_CHAT_PROMPT_TIMEOUT_SECONDS", "2.0"))
                        except Exception:
                            _timeout_s = 2.0
                        try:
                            ctx = await asyncio.wait_for(_build_classifier_context_summary(sid), timeout=_timeout_s)
                            if ctx:
                                ctx_source = "server"
                        except Exception:
                            ctx = ""
                            ctx_source = "none"
                        # Enrich with profile + goals if available
                        try:
                            # Check for mock user data first (passed from frontend in mock mode)
                            mock_profile = None
                            mock_goals = None
                            if mock_user_data:
                                try:
                                    s = str(mock_user_data)
                                    pad = "=" * (-len(s) % 4)
                                    mock_decoded = base64.urlsafe_b64decode((s + pad).encode("utf-8")).decode("utf-8", errors="ignore")
                                    mock_data = json.loads(mock_decoded)
                                    mock_profile = mock_data.get("profile")
                                    mock_goals = mock_data.get("goals", [])
                                    logger.info(f"[DEBUG] Using mock user data: profile={bool(mock_profile)}, goals={len(mock_goals) if mock_goals else 0}")
                                except Exception as e:
                                    logger.error(f"[DEBUG] Failed to decode mock user data: {e}")

                            # Resolve userId from session (use mock data if available, otherwise query Convex)
                            uid = None
                            if mock_profile:
                                # Use mock data
                                uid = mock_profile.get("userId")
                            else:
                                # Query Convex
                                doc = await _get_session_doc(str(sid))
                                uid = (doc or {}).get("userId") if isinstance(doc, dict) else None

                            if uid:
                                prof = mock_profile if mock_profile else await _get_user_profile(str(uid))
                                if isinstance(prof, dict):
                                    bio = str(prof.get("bio") or "").strip()
                                    name = str(prof.get("displayName") or "").strip()
                                    user_profile_text = (f"User profile: {name}. Bio: {bio}" if bio or name else "")

                                goals = mock_goals if mock_goals else await _list_user_goals(str(uid))
                                if isinstance(goals, list) and goals:
                                    lines = []
                                    for g in goals[:5]:
                                        t = str((g or {}).get("title") or "").strip()
                                        st = str((g or {}).get("status") or "").strip()
                                        if t:
                                            lines.append(f"- {t}{(' ['+st+']') if st else ''}")
                                    if lines:
                                        goals_text = "User goals:\n" + "\n".join(lines)
                        except Exception:
                            user_profile_text = ""
                            goals_text = ""
                    infused_ctx = ctx
                    if user_profile_text or goals_text:
                        ext = "\n\n".join(s for s in [user_profile_text, goals_text] if s)
                        infused_ctx = (f"{ctx}\n\n{ext}" if ctx else ext)
                    infused = (f"ctx: {infused_ctx}\nmsg: {prompt or ''}" if infused_ctx else (prompt or ""))
                    
                    # Build system prompt with speech coaching context and tracked skills (with timeout)
                    try:
                        _timeout_s = float(os.getenv("AI_CHAT_PROMPT_TIMEOUT_SECONDS", "2.0"))
                    except Exception:
                        _timeout_s = 2.0
                    try:
                        # Prefer client-provided tracked skills when available (base64url JSON array)
                        client_skills: Optional[List[Dict[str, Any]]] = None
                        try:
                            skills_b64 = request.query_params.get("skills")
                            if skills_b64:
                                s = str(skills_b64)
                                pad = "=" * (-len(s) % 4)
                                decoded = base64.urlsafe_b64decode((s + pad).encode("utf-8")).decode("utf-8", errors="ignore")
                                arr = json.loads(decoded)
                                if isinstance(arr, list) and arr:
                                    norm: List[Dict[str, Any]] = []
                                    for it in arr:
                                        if isinstance(it, dict):
                                            obj = {
                                                "id": it.get("id"),
                                                "name": it.get("name") or it.get("title"),
                                                "category": it.get("category"),
                                            }
                                            # Drop empties
                                            obj = {k: v for k, v in obj.items() if v}
                                            norm.append(obj)
                                        elif isinstance(it, str):
                                            norm.append({"name": it})
                                    if norm:
                                        client_skills = norm
                        except Exception:
                            client_skills = None

                        # Extract user profile and goals from request parameters
                        user_profile: Optional[Dict[str, Any]] = None
                        user_goals: Optional[List[Dict[str, Any]]] = None

                        # Extract user profile (base64url JSON object)
                        try:
                            profile_b64 = request.query_params.get("userProfile")
                            if profile_b64:
                                s = str(profile_b64)
                                pad = "=" * (-len(s) % 4)
                                decoded = base64.urlsafe_b64decode((s + pad).encode("utf-8")).decode("utf-8", errors="ignore")
                                profile_data = json.loads(decoded)
                                if isinstance(profile_data, dict):
                                    user_profile = profile_data
                        except Exception:
                            user_profile = None

                        # Extract user goals (base64url JSON array)
                        try:
                            goals_b64 = request.query_params.get("userGoals")
                            if goals_b64:
                                s = str(goals_b64)
                                pad = "=" * (-len(s) % 4)
                                decoded = base64.urlsafe_b64decode((s + pad).encode("utf-8")).decode("utf-8", errors="ignore")
                                goals_data = json.loads(decoded)
                                if isinstance(goals_data, list):
                                    user_goals = goals_data
                        except Exception:
                            user_goals = None

                        system_prompt = await asyncio.wait_for(
                            _build_system_prompt(
                                sid,
                                client_skills=client_skills,
                                user_profile=user_profile,
                                user_goals=user_goals
                            ),
                            timeout=_timeout_s
                        )
                    except Exception:
                        system_prompt = ""
                    
                    # Debug: log exact prompt payload being sent to chat LLM (with context details)
                    try:
                        # Always log lightweight, redacted metadata
                        logger.info(
                            json.dumps(
                                {
                                    "event": "chat_stream_request",
                                    "requestId": request_id,
                                    "trackedSkillIdHash": tracked_hash,
                                    "route": "/chat/stream",
                                    "provider": provider_name,
                                    "model": model_name,
                                    "prompt_len": len(prompt or ""),
                                    "ctx_present": bool(ctx),
                                    "ctx_source": ctx_source,
                                    "history_present": bool(hist_b64),
                                    "ctx_len": len(ctx or ""),
                                    "infused_len": len(infused),
                                    "system_prompt_len": len(system_prompt),
                                }
                            )
                        )
                        # Verbose content logging only when explicitly enabled
                        _dbg_logs = str(os.getenv("AI_CHAT_DEBUG_LOGS", "")).strip().lower() in ("1", "true", "yes", "on")
                        if _dbg_logs:
                            try:
                                logger.info(f"[DEBUG] AI_CHAT_ENABLED={_enabled}, AI_CHAT_MODEL={model_name}")
                                logger.info(f"[DEBUG] System prompt: {system_prompt}")
                                logger.info(f"[DEBUG] User message (infused): {infused}")
                            except Exception:
                                pass
                    except Exception:
                        pass

                    async def _provider_stream():
                        nonlocal first_token_s
                        # Emit prompt event for debugging when requested
                        try:
                            dbg = str(request.query_params.get("debug") or "").strip() in ("1", "true", "yes", "on") or str(os.getenv("AI_CHAT_PROMPT_EVENT", "")).strip().lower() in ("1", "true", "yes", "on")
                        except Exception:
                            dbg = False
                        if dbg:
                            try:
                                rendered = (f"{system_prompt}\n\n{infused}" if system_prompt else infused)
                                payload = {"system": system_prompt, "user": infused, "rendered": rendered, "ctx_source": ctx_source}
                                yield f"event: prompt\n"
                                yield f"data: {json.dumps(payload)}\n\n"
                            except Exception:
                                pass
                        # Implement TTFT timeout: if the first token takes too long, fall back to stub
                        try:
                            try:
                                _ttft_timeout_s = float(os.getenv("AI_CHAT_TTFT_TIMEOUT_SECONDS", "5.0"))
                            except Exception:
                                _ttft_timeout_s = 5.0

                            agen = client.stream_chat(infused, system=system_prompt, request_id=request_id).__aiter__()
                            try:
                                first_token = await asyncio.wait_for(agen.__anext__(), timeout=_ttft_timeout_s)
                            except asyncio.TimeoutError:
                                # Timeout waiting for first token — fall back to stub stream
                                try:
                                    logger.info(
                                        json.dumps(
                                            {
                                                "event": "chat_stream_ttft_timeout",
                                                "requestId": request_id,
                                                "trackedSkillIdHash": tracked_hash,
                                                "route": "/chat/stream",
                                                "timeout_ms": int(_ttft_timeout_s * 1000),
                                                "provider": provider_name,
                                                "model": model_name,
                                            }
                                        )
                                    )
                                except Exception:
                                    pass
                                # Emit a non-audible comment notice and switch to stub
                                yield f": provider timeout after {int(_ttft_timeout_s)}s, falling back\n\n"
                                async for chunk in _stub_stream():
                                    yield chunk
                                return
                            except StopAsyncIteration:
                                # Provider finished without tokens — treat as empty and finish
                                yield "data: [DONE]\n\n"
                                return

                            # We got the first token — record metrics and yield it
                            chunk = _sse_data_event(first_token)
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
                                # Metrics: observe TTFT for provider path
                                try:
                                    CHAT_TTFT_SECONDS.labels(
                                        provider=str(provider_name or "provider"),
                                        model=str(model_name or "unknown"),
                                    ).observe(first_token_s)
                                except Exception:
                                    pass
                            yield chunk

                            # Stream the rest
                            async for token in agen:
                                yield _sse_data_event(token)
                            # End-of-stream marker
                            yield "data: [DONE]\n\n"
                        except Exception as e:
                            # Any runtime error: notify client then fall back to stub stream
                            try:
                                logger.exception(
                                    json.dumps(
                                        {
                                            "event": "chat_stream_provider_runtime_error",
                                            "requestId": request_id,
                                            "trackedSkillIdHash": tracked_hash,
                                            "route": "/chat/stream",
                                            "provider": provider_name,
                                            "model": model_name,
                                            "error": str(e),
                                        }
                                    )
                                )
                            except Exception:
                                pass
                            # Emit a non-audible comment with the error summary
                            yield f": provider runtime error: {str(e)[:240]}\n\n"
                            async for chunk in _stub_stream():
                                yield chunk

                    _token_stream = _provider_stream
                except Exception as e:
                    # Log provider error and fall back to stub stream
                    try:
                        logger.exception(
                            json.dumps(
                                {
                                    "event": "chat_stream_provider_error",
                                    "requestId": request_id,
                                    "trackedSkillIdHash": tracked_hash,
                                    "route": "/chat/stream",
                                    "prompt_present": bool(prompt),
                                    "provider": provider_name or "unknown",
                                    "model": model_name,
                                    "error": str(e),
                                }
                            )
                        )
                    except Exception:
                        pass
                    provider_setup_error = str(e)
                    # Fallback to stub stream on any provider error
                    provider_name = provider_name or "fallback_stub"
                    model_name = model_name or None

            if _token_stream is None:
                # Stub mode: return a simple response
                provider_name = provider_name or "stub"
                model_name = None
                
                # Build context and system prompt even in stub mode for debugging
                sid = session_id or request.query_params.get("session_id") or request.query_params.get("sessionId")
                ctx = ""
                ctx_source = "none"
                hist_b64 = request.query_params.get("history")
                # Try client-provided history (base64url JSON array of {role, content})
                if hist_b64:
                    try:
                        s = str(hist_b64)
                        pad = "=" * (-len(s) % 4)
                        decoded = base64.urlsafe_b64decode((s + pad).encode("utf-8")).decode("utf-8", errors="ignore")
                        arr = json.loads(decoded)
                        if isinstance(arr, list) and arr:
                            # Limit number of messages for safety
                            try:
                                limit = _chat_context_limit_default()
                            except Exception:
                                limit = 10
                            items = arr[-max(1, min(200, limit)) :]
                            lines: List[str] = []
                            lines_turns: List[str] = []
                            lines_unknown: List[str] = []
                            # Env-configurable caps
                            _item_word_cap = _chat_item_word_cap_default()
                            _total_word_cap = _chat_total_word_cap_default()
                            _item_char_cap = _chat_item_char_cap_default()
                            _total_char_cap = _chat_total_char_cap_default()
                            for it in items:
                                if not isinstance(it, dict):
                                    continue
                                role = str((it.get("role") or "")).lower()
                                tag = "u" if role == "user" else ("a" if role == "assistant" else "?")
                                txt_raw = str((it.get("content") or "")).replace("\n", " ").strip()
                                # Per-item truncation: prefer words, fallback to chars
                                if _item_word_cap > 0:
                                    txt = _truncate_by_words(txt_raw, _item_word_cap)
                                else:
                                    _cap = max(1, int(_item_char_cap or 240))
                                    txt = (txt_raw[:max(0, _cap - 3)] + "...") if len(txt_raw) > _cap else txt_raw
                                if txt:
                                    line = f"{tag}: {txt}"
                                    lines.append(line)
                                    if tag == "?":
                                        lines_unknown.append(line)
                                    else:
                                        lines_turns.append(line)
                            # Present context in sections for readability
                            sections: List[str] = []
                            if lines_unknown:
                                unknown_fmt = ["- " + l[3:] if l.startswith("?: ") else ("- " + l) for l in lines_unknown]
                                sections.append("Session summary:\n" + "\n".join(unknown_fmt))
                            if lines_turns:
                                turns_fmt = ["- " + l for l in lines_turns]
                                sections.append("Recent turns:\n" + "\n".join(turns_fmt))
                            ctx = "\n\n".join(sections) if sections else "\n".join(["- " + l for l in lines])
                            # Total truncation: prefer words, fallback to chars
                            if _total_word_cap > 0:
                                ctx = _truncate_by_words(ctx, _total_word_cap)
                            else:
                                _tcap = max(1, int(_total_char_cap or 6000))
                                if len(ctx) > _tcap:
                                    ctx = ctx[:max(0, _tcap - 3)] + "..."
                            if ctx:
                                ctx_source = "client"
                    except Exception:
                        pass
                # Fallback to server-side transcript summary when client history not present/invalid (with timeout)
                if not ctx and sid:
                    try:
                        _timeout_s = float(os.getenv("AI_CHAT_PROMPT_TIMEOUT_SECONDS", "2.0"))
                    except Exception:
                        _timeout_s = 2.0
                    try:
                        ctx = await asyncio.wait_for(_build_classifier_context_summary(sid), timeout=_timeout_s)
                        if ctx:
                            ctx_source = "server"
                    except Exception:
                        ctx = ""
                        ctx_source = "none"
                infused = (f"ctx: {ctx}\nmsg: {prompt or ''}" if ctx else (prompt or ""))
                
                # Build system prompt with speech coaching context and tracked skills (with timeout)
                try:
                    _timeout_s = float(os.getenv("AI_CHAT_PROMPT_TIMEOUT_SECONDS", "2.0"))
                except Exception:
                    _timeout_s = 2.0
                try:
                    system_prompt = await asyncio.wait_for(_build_system_prompt(sid), timeout=_timeout_s)
                except Exception:
                    system_prompt = ""
                
                # Optional verbose content logging (guarded to avoid PII in non-dev)
                try:
                    _dbg_logs = str(os.getenv("AI_CHAT_DEBUG_LOGS", "")).strip().lower() in ("1", "true", "yes", "on")
                    if _dbg_logs:
                        logger.info(f"[DEBUG] System prompt: {system_prompt}")
                        logger.info(f"[DEBUG] User message (infused): {infused}")
                except Exception:
                    pass
                
                async def _stub_with_reason():
                    # Emit prompt event for debugging when requested
                    try:
                        dbg = str(request.query_params.get("debug") or "").strip() in ("1", "true", "yes", "on") or str(os.getenv("AI_CHAT_PROMPT_EVENT", "")).strip().lower() in ("1", "true", "yes", "on")
                    except Exception:
                        dbg = False
                    if dbg:
                        try:
                            rendered = (f"{system_prompt}\n\n{infused}" if system_prompt else infused)
                            payload = {
                                "system": system_prompt or _system_prompt_base(),
                                "user": infused,
                                "rendered": (f"{_system_prompt_base()}\n\n{infused}") if not system_prompt else rendered,
                                "ctx_source": ctx_source,
                            }
                            yield f"event: prompt\n"
                            yield f"data: {json.dumps(payload)}\n\n"
                        except Exception:
                            pass
                    if dbg:
                        try:
                            rendered = (f"{system_prompt}\n\n{infused}" if system_prompt else infused)
                            payload = {
                                "system": system_prompt or _system_prompt_base(),
                                "user": infused,
                                "rendered": (f"{_system_prompt_base()}\n\n{infused}") if not system_prompt else rendered,
                                "ctx_source": ctx_source,
                            }
                            yield f"event: prompt\n"
                            yield f"data: {json.dumps(payload)}\n\n"
                        except Exception:
                            pass
                    if provider_setup_error:
                        yield f"data: [provider setup error: {provider_setup_error}]\n\n"
                        await asyncio.sleep(0.01)
                    async for chunk in _stub_stream():
                        yield chunk

                _token_stream = _stub_with_reason
            # Heartbeat wrapper to keep long-lived SSE connections alive behind proxies
            try:
                _hb_env = os.getenv("AI_CHAT_SSE_HEARTBEAT_SECONDS", "15")
                _hb_s = float(_hb_env) if _hb_env is not None else 15.0
            except Exception:
                _hb_s = 15.0

            async def _with_heartbeat(agen, interval_s: float = _hb_s):
                """Yield from agen but emit SSE comment heartbeats every interval when idle."""
                nxt_task = asyncio.create_task(agen.__anext__())
                try:
                    while True:
                        sleep_task = asyncio.create_task(asyncio.sleep(max(0.001, float(interval_s))))
                        done, pending = await asyncio.wait({nxt_task, sleep_task}, return_when=asyncio.FIRST_COMPLETED)
                        if nxt_task in done:
                            try:
                                chunk = nxt_task.result()
                            except StopAsyncIteration:
                                break
                            # Schedule next token fetch
                            nxt_task = asyncio.create_task(agen.__anext__())
                            yield chunk
                        else:
                            # Heartbeat comment (SSE ignores comment lines)
                            yield ": ping\n\n"
                            # Continue waiting on the same next-token task
                finally:
                    try:
                        nxt_task.cancel()
                    except Exception:
                        pass

            async for chunk in _with_heartbeat(_token_stream()):
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
                    # Metrics: observe TTFT for fallback/stub path
                    try:
                        CHAT_TTFT_SECONDS.labels(
                            provider=str(provider_name or ("provider" if _enabled else "stub")),
                            model=str(model_name or "unknown"),
                        ).observe(first_token_s)
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
            # Metrics: observe total stream duration
            try:
                CHAT_TOTAL_SECONDS.labels(
                    provider=str(provider_name or ("provider" if _enabled else "stub")),
                    model=str(model_name or "unknown"),
                ).observe(total_s)
            except Exception:
                pass

    # Ensure header values are sane even before the stream starts
    if not _enabled and not provider_name:
        provider_name = "stub"
        model_name = None
    resp = StreamingResponse(instrumented_stream(), media_type="text/event-stream; charset=utf-8")
    if request_id:
        # Echo for clients/proxies that want to surface it
        resp.headers["X-Request-Id"] = str(request_id)
    # Surface provider/model so benchmarks and clients can record them
    if provider_name:
        resp.headers["X-Chat-Provider"] = str(provider_name)
    if model_name:
        resp.headers["X-Chat-Model"] = str(model_name)
    # SSE anti-buffering headers
    resp.headers["Cache-Control"] = "no-cache, no-transform"
    resp.headers["Connection"] = "keep-alive"
    # Disable nginx proxy buffering if present
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.post("/api/v1/tts", tags=["audio"], description="Stub TTS that returns a short WAV file URL for benchmarking.")
async def tts_generate(
    request: Request,
    body: Dict[str, Any] = Body(..., description="JSON body with 'text' and optional 'sessionId'"),
):
    """Generate a short silent WAV and return a URL to fetch it.

    This is a lightweight stub to unblock e2e benchmarks. It does not call a real TTS provider.
    """
    try:
        text = str(body.get("text", ""))
    except Exception:
        text = ""
    # If configured, proxy to the frontend Next.js TTS route for real synthesis
    # Env:
    # - TTS_PROXY_URL: e.g. http://localhost:3000/api/v1/tts
    # - TTS_PROXY_AUTH: optional Authorization header value (e.g. "Bearer ...")
    proxy_url = os.getenv("TTS_PROXY_URL")
    if proxy_url:
        try:
            headers = {"content-type": "application/json"}
            # Propagate request id when available for tracing
            req_id = getattr(getattr(request, "state", object()), "request_id", None) or request.headers.get("x-request-id")
            if req_id:
                headers["x-request-id"] = str(req_id)
            auth_header = os.getenv("TTS_PROXY_AUTH")
            if auth_header:
                headers["authorization"] = auth_header
            payload = {
                "text": text,
                # Pass through optional fields if provided by caller
                "voiceId": body.get("voiceId"),
                "format": body.get("format"),
                "sessionId": body.get("sessionId"),
                "groupId": body.get("groupId"),
                # Allow provider override passthrough when frontend permits it
                "provider": body.get("provider"),
            }
            # Remove None to keep payload clean
            payload = {k: v for k, v in payload.items() if v is not None}
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(str(proxy_url), json=payload, headers=headers)
            if resp.status_code == 200:
                # Return the frontend payload directly
                return JSONResponse(resp.json())
            else:
                # Fall back to stub on non-200, but include proxy error info
                try:
                    err_text = resp.text
                except Exception:
                    err_text = ""
                logging.getLogger("coach_up.ai.tts").warning(
                    "tts_proxy_non_200 status=%s body=%s", resp.status_code, err_text[:512]
                )
        except Exception as e:
            # Fall back to stub on proxy failure
            try:
                logging.getLogger("coach_up.ai.tts").exception("tts_proxy_error: %s", e)
            except Exception:
                pass

    # Determine duration based on text length (bounded)
    try:
        base_s = 0.6
        add_s = min(2.4, max(0.0, len(text) * 0.01))  # 10ms per char, capped
        duration_s = max(0.5, min(3.0, base_s + add_s))
    except Exception:
        duration_s = 1.0

    # Generate a mono 16-bit PCM WAV of silence at 16kHz
    fr = 16000
    sampwidth = 2
    nframes = int(duration_s * fr)
    buf = io.BytesIO()
    try:
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(sampwidth)
            w.setframerate(fr)
            # Write silence
            w.writeframes(b"\x00\x00" * nframes)
        audio_bytes = buf.getvalue()
    finally:
        buf.close()

    # Store in-memory and return a fetchable URL
    audio_id = str(uuid.uuid4())
    try:
        store: Dict[str, bytes] = app.state.tts_audio  # type: ignore[attr-defined]
    except Exception:
        store = {}
        try:
            app.state.tts_audio = store  # type: ignore[attr-defined]
        except Exception:
            pass
    store[audio_id] = audio_bytes

    # Absolute URL for the audio resource
    try:
        audio_url = str(request.url_for("tts_audio", audio_id=audio_id))
    except Exception:
        # Fallback to relative path
        audio_url = f"/api/v1/tts/audio/{audio_id}.wav"

    return JSONResponse({
        "audioUrl": audio_url,
        "durationMs": int(duration_s * 1000),
        "provider": "stub",
    })


@app.get(
    "/api/v1/tts/audio/{audio_id}.wav",
    tags=["audio"],
    name="tts_audio",
    description="Serve previously generated stub TTS WAV bytes.",
)
async def tts_audio(audio_id: str):
    try:
        store: Dict[str, bytes] = app.state.tts_audio  # type: ignore[attr-defined]
    except Exception:
        store = {}
    data = store.get(audio_id)
    if not data:
        return Response(status_code=404)
    return Response(content=data, media_type="audio/wav")

def _compute_rubric_v1_summary_from_spans(
    session_id: str,
    group_id: str,
    scores: Dict[str, float],
    rubric_version: str = "v2",
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

@app.get(
    "/api/v1/session-summary",
    tags=["chat"],
    description="Return latest rolling summary for a session (independent of assessments).",
)
async def get_session_summary(sessionId: Optional[str] = Query(None, description="Session ID")):
    if not sessionId:
        return JSONResponse({"version": 0, "updatedAt": 0, "summaryText": ""}, status_code=200)
    try:
        store: Dict[str, Dict[str, Any]] = app.state.session_summaries  # type: ignore[attr-defined]
    except Exception:
        store = {}
    row = store.get(sessionId) or {"version": 0, "updatedAt": 0, "summaryText": ""}
    etag = f"{sessionId}:{row.get('version', 0)}"
    headers = {"ETag": etag}
    return JSONResponse(row, headers=headers, status_code=200)


@app.post(
    "/api/v1/session-summary/generate",
    tags=["chat"],
    description="Generate a rolling session summary using the configured provider (e.g., Google Gemini).",
)
async def post_session_summary_generate(
    request: Request,
    body: Dict[str, Any] = Body(..., description="{ sessionId, prevSummary?, messages?, tokenBudget? }"),
):
    sid = str((body or {}).get("sessionId") or "").strip()
    prev = str((body or {}).get("prevSummary") or "")
    messages = (body or {}).get("messages") or (body or {}).get("recentMessages") or []
    token_budget = (body or {}).get("tokenBudget")
    try:
        token_budget = int(token_budget) if token_budget is not None else None
    except Exception:
        token_budget = None
    # Enforce a backend minimum/default token budget to avoid MAX_TOKENS empty outputs
    try:
        _min_budget = int(os.getenv("AI_SUMMARY_TOKEN_BUDGET_MIN", "1200").strip())
    except Exception:
        _min_budget = 1200
    token_budget_final: Optional[int]
    if token_budget is None:
        token_budget_final = _min_budget
    else:
        token_budget_final = max(_min_budget, int(token_budget))
    if not sid:
        return JSONResponse({"error": "sessionId is required"}, status_code=400)

    t0 = time.perf_counter()
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    try:
        # Debug: log incoming request shape for session summary generation
        try:
            logger.info(json.dumps({
                "event": "summary_generate_request",
                "sessionId": sid,
                "prevLen": len(prev or ""),
                "messagesCount": len(list(messages) if isinstance(messages, list) else []),
                "tokenBudget": token_budget_final,
                "requestId": req_id,
            }))
        except Exception:
            pass
        # Optional: emit a safe sample of inputs for diagnosis
        try:
            _dbg = str(os.getenv("AI_SUMMARY_DEBUG_LOGS", "")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            _dbg = False
        if _dbg:
            try:
                msgs = list(messages) if isinstance(messages, list) else []
                sample: List[Dict[str, str]] = []
                for m in (msgs[:2] + msgs[-2:] if len(msgs) > 4 else msgs):
                    try:
                        role = str((m.get("role") or ""))
                        content = str((m.get("content") or "")).replace("\n", " ")
                        if len(content) > 200:
                            content = content[:197] + "..."
                        sample.append({"role": role, "content": content})
                    except Exception:
                        continue
                prev_preview = (prev or "").replace("\n", " ")
                if len(prev_preview) > 240:
                    prev_preview = prev_preview[:237] + "..."
                logger.info(json.dumps({
                    "event": "summary_generate_input_sample",
                    "requestId": req_id,
                    "sessionId": sid,
                    "prevPreview": prev_preview,
                    "sampleMessages": sample,
                }))
            except Exception:
                pass

        client = get_summary_client()
        text = await client.summarize(prev, list(messages), token_budget=token_budget_final, request_id=req_id)  # type: ignore[arg-type]
        # Access in-memory store and current row
        try:
            store: Dict[str, Dict[str, Any]] = app.state.session_summaries  # type: ignore[attr-defined]
        except Exception:
            store = {}
        row = store.get(sid)
        prev_version = int((row or {}).get("version") or 0)
        prev_updated = int((row or {}).get("updatedAt") or 0)
        # If model returned empty text, make it explicit and avoid persisting empty
        if not (text or "").strip():
            try:
                logger.info(json.dumps({
                    "event": "summary_generate_empty",
                    "provider": getattr(client, "provider_name", "unknown"),
                    "model": getattr(client, "model", None),
                    "sessionId": sid,
                    "prevVersion": prev_version,
                    "requestId": req_id,
                }))
            except Exception:
                pass
            payload = {"sessionId": sid, "status": "empty", "version": prev_version, "updatedAt": prev_updated, "text": ""}
            return JSONResponse(payload, status_code=200, headers={"X-Summary-Empty": "1"})
        # Persist to in-memory store for immediate GET visibility
        version = prev_version + 1
        now_ms = int(time.time() * 1000)
        out = {"version": version, "updatedAt": now_ms, "summaryText": text}
        store[sid] = out
        try:
            logger.info(json.dumps({
                "event": "summary_generate",
                "provider": getattr(client, "provider_name", "unknown"),
                "model": getattr(client, "model", None),
                "sessionId": sid,
                "len": len(text or ""),
                "durationMs": int((time.perf_counter() - t0) * 1000),
                "requestId": req_id,
            }))
        except Exception:
            pass
        # Optional verbose content logging (guarded by env)
        try:
            _dbg = str(os.getenv("AI_SUMMARY_DEBUG_LOGS", "")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            _dbg = False
        if _dbg:
            try:
                logger.info(f"[DEBUG] Summary prev len={len(prev or '')}, messages={len(list(messages))}, generated len={len(text or '')}")
                preview = (text or "")[:300].replace("\n", " ")
                logger.info(f"[DEBUG] Summary preview: {preview}")
            except Exception:
                pass
        return JSONResponse({"sessionId": sid, "version": version, "updatedAt": now_ms, "text": text}, status_code=200)
    except Exception as e:
        try:
            logger.error(json.dumps({
                "event": "summary_generate_error",
                "sessionId": sid,
                "error": type(e).__name__,
                "message": str(e)[:512],
                "requestId": req_id,
            }))
        except Exception:
            pass
        return JSONResponse({"error": "summary generate failed"}, status_code=502)

def _system_prompt_base() -> str:
    return (
        "You are a concise, friendly speech coach. Your purpose is to help users improve speaking skills through short, actionable guidance and practice.\n\n"
        "Behavioral rules:\n"
        "- Greetings and small talk: reply with 1 short friendly sentence, then immediately pivot to the goal.\n"
        "- Do not analyze pleasantries or single-line greetings.\n"
        "- Default response length: at most 2–3 sentences or 5 short bullets.\n"
        "- Ask exactly one question to clarify goals or select the next focus area.\n"
        "- Only analyze speech when the user asks for analysis/feedback or after they choose a focus area.\n"
        "- If the user hasn’t specified a focus, offer 3–5 options.\n"
        "- When providing guidance, prefer practical, immediately applicable tips.\n"
        "- Avoid repeating the user’s message. Avoid long lectures. Focus on short, actionable advice. Be supportive and encouraging.\n"
        "If a focus area is chosen:\n"
        "- Suggest a short excercise to gauge progress, and then focus on one technique at a time that will provide the most value.\n"
        "If user asks for analysis:\n"
        "- Keep analysis brief (2–3 bullets), then 1–2 concrete next actions.\n\n"
        "Stay conversational, positive, and time-efficient."
    )


async def _build_system_prompt(
    session_id: Optional[str],
    *,
    client_skills: Optional[List[Dict[str, Any]]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    user_goals: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build system prompt with speech coaching context, user profile, goals, and tracked skills."""
    base_prompt = _system_prompt_base()

    # Add user profile context if available
    if user_profile and isinstance(user_profile, dict):
        display_name = str(user_profile.get("displayName") or "").strip()
        bio = str(user_profile.get("bio") or "").strip()
        if display_name or bio:
            profile_parts = []
            if display_name:
                profile_parts.append(f"Name: {display_name}")
            if bio:
                profile_parts.append(f"Bio: {bio}")
            if profile_parts:
                profile_text = " | ".join(profile_parts)
                base_prompt += f"User Profile: {profile_text}\n"

    # Add user goals context if available
    if user_goals and isinstance(user_goals, list) and user_goals:
        goal_descriptions = []
        for goal in user_goals:
            if isinstance(goal, dict):
                title = str(goal.get("title") or "").strip()
                description = str(goal.get("description") or "").strip()
                if title:
                    if description:
                        goal_descriptions.append(f"{title}: {description}")
                    else:
                        goal_descriptions.append(title)
        if goal_descriptions:
            goals_text = "; ".join(goal_descriptions)
            base_prompt += f"User Goals: {goals_text}\n"

    # Add tracked skills context if available
    skills: List[Dict[str, Any]] = []
    # Prefer client-provided skills to avoid extra round-trips
    if client_skills and isinstance(client_skills, list):
        skills = client_skills
    elif session_id:
        try:
            skills = await _get_tracked_skills_for_session(session_id)
        except Exception:
            skills = []
    if skills:
        skill_names = [str(s.get("name") or s.get("title") or s.get("id") or "").strip() for s in skills]
        skill_names = [n for n in skill_names if n]
        if skill_names:
            try:
                print(f"[DEBUG] Tracked skills resolved for session {session_id}: {skill_names}")
                logger.info(
                    json.dumps(
                        {
                            "event": "tracked_skills_resolved",
                            "route": "/chat/stream",
                            "session_id_present": bool(session_id),
                            "skill_count": len(skill_names),
                            "skill_names": skill_names,
                            "source": "client" if client_skills else "server",
                        }
                    )
                )
            except Exception:
                pass
            skills_text = ", ".join(skill_names)
            base_prompt += (
                f"Current focus areas for this user: {skills_text}\n"
                "These reflect the user's selected goals. Prioritize these areas when providing tips, but keep responses brief.\n\n"
            )
    base_prompt += (
        "Respond naturally and conversationally. Keep responses concise but helpful. "
        "Focus on practical advice the user can immediately apply."
    )

    return base_prompt

@app.get("/chat/tracked-skills", tags=["chat"], description="Fetch tracked skills for a session (for client caching).")
async def chat_tracked_skills(session_id: Optional[str] = Query(None, description="Session ID")):
    try:
        if not session_id:
            return JSONResponse({"skills": []}, status_code=200)
        skills = await _get_tracked_skills_for_session(session_id)
        # Normalize to minimal shape
        out: List[Dict[str, Any]] = []
        for s in skills or []:
            if not isinstance(s, dict):
                continue
            obj = {
                "id": s.get("id"),
                "name": s.get("name") or s.get("title"),
                "category": s.get("category"),
            }
            obj = {k: v for k, v in obj.items() if v}
            if obj.get("name"):
                out.append(obj)
        return JSONResponse({"skills": out}, status_code=200)
    except Exception as e:
        try:
            logger.exception("/chat/tracked-skills error: %s", e)
        except Exception:
            pass
        return JSONResponse({"skills": []}, status_code=200)

async def _get_tracked_skills_for_session(session_id: str) -> List[Dict[str, Any]]:
    """Fetch tracked skills for a session.

    Flow:
      1) Resolve userId from Convex via sessions:getBySessionId
      2) Fetch tracked skills for that user via skills:getTrackedSkillsForUser
      3) Normalize to [{id,name,category}] ordered by tracked order
      4) On any failure or missing config, fall back to ASSESS_TRACKED_SKILLS_JSON
    """

    def _fallback_from_env() -> List[Dict[str, Any]]:
        try:
            raw = (os.getenv("ASSESS_TRACKED_SKILLS_JSON", "") or "").strip()
            if not raw:
                # If MOCK_CONVEX is enabled, seed a default set of skills
                mock_on = (os.getenv("MOCK_CONVEX", "0").strip().lower() in ("1", "true", "yes", "on"))
                if mock_on:
                    seeded = [
                        {"id": "clarity", "name": "Clarity", "category": "delivery"},
                        {"id": "pacing", "name": "Pacing", "category": "delivery"},
                    ]
                    try:
                        print("[DEBUG] MOCK_CONVEX enabled; seeding tracked skills fallback")
                        logger.info(
                            json.dumps(
                                {
                                    "event": "tracked_skills_mock_seeded",
                                    "count": len(seeded),
                                    "names": [s["name"] for s in seeded],
                                }
                            )
                        )
                    except Exception:
                        pass
                    return seeded
                else:
                    try:
                        print("[DEBUG] No ASSESS_TRACKED_SKILLS_JSON present; returning empty tracked skills fallback")
                        logger.info("[DEBUG] No ASSESS_TRACKED_SKILLS_JSON present; returning empty tracked skills fallback")
                    except Exception:
                        pass
                    return []
            val = json.loads(raw)
            if isinstance(val, list):
                out2: List[Dict[str, Any]] = []
                for it in val:
                    if isinstance(it, dict):
                        out2.append(
                            {
                                "id": it.get("id"),
                                "name": it.get("name"),
                                "category": it.get("category"),
                            }
                        )
                try:
                    names = [d.get("name") for d in out2 if d.get("name")]
                    print(f"[DEBUG] Using tracked skills from ASSESS_TRACKED_SKILLS_JSON: {names}")
                    logger.info(
                        json.dumps(
                            {
                                "event": "tracked_skills_fallback_env",
                                "count": len(names),
                                "names": names,
                            }
                        )
                    )
                except Exception:
                    pass
                return out2
        except Exception:
            pass
        return []

    try:
        base_url = (os.getenv("CONVEX_URL") or "").strip()
        if not base_url:
            try:
                print("[DEBUG] CONVEX_URL not set; falling back to ASSESS_TRACKED_SKILLS_JSON")
                logger.info("[DEBUG] CONVEX_URL not set; falling back to ASSESS_TRACKED_SKILLS_JSON")
            except Exception:
                pass
            return _fallback_from_env()

        url = base_url.rstrip("/") + "/api/query"
        # Reuse global HTTP timeout when present; default 10s
        try:
            timeout_s = float(os.getenv("CONVEX_TIMEOUT_SECONDS") or os.getenv("AI_HTTP_TIMEOUT_SECONDS") or 10)
        except Exception:
            timeout_s = 10.0

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            # 1) Resolve session -> userId
            payload1 = {
                "path": "functions/sessions:getBySessionId",
                "args": {"sessionId": session_id},
                "format": "json",
            }
            resp1 = await client.post(url, json=payload1, headers={"Content-Type": "application/json"})
            data1 = resp1.json() if resp1.headers.get("content-type", "").startswith("application/json") else {}
            if resp1.status_code >= 400 or (isinstance(data1, dict) and data1.get("status") == "error"):
                try:
                    print(f"[DEBUG] Convex sessions:getBySessionId error status={resp1.status_code}; falling back to env")
                    logger.info("[DEBUG] Convex sessions:getBySessionId error; falling back to env")
                except Exception:
                    pass
                return _fallback_from_env()
            session_doc = data1.get("value") if isinstance(data1, dict) else None
            user_id = (session_doc or {}).get("userId") if isinstance(session_doc, dict) else None
            if not user_id or not str(user_id).strip():
                try:
                    print("[DEBUG] No userId resolved from session; falling back to env")
                    logger.info("[DEBUG] No userId resolved from session; falling back to env")
                except Exception:
                    pass
                return _fallback_from_env()

            # 2) Fetch tracked skills for user
            payload2 = {
                "path": "functions/skills:getTrackedSkillsForUser",
                "args": {"userId": str(user_id)},
                "format": "json",
            }
            resp2 = await client.post(url, json=payload2, headers={"Content-Type": "application/json"})
            data2 = resp2.json() if resp2.headers.get("content-type", "").startswith("application/json") else {}
            if resp2.status_code >= 400 or (isinstance(data2, dict) and data2.get("status") == "error"):
                try:
                    print(f"[DEBUG] Convex skills:getTrackedSkillsForUser error status={resp2.status_code}; falling back to env")
                    logger.info("[DEBUG] Convex skills:getTrackedSkillsForUser error; falling back to env")
                except Exception:
                    pass
                return _fallback_from_env()
            rows = data2.get("value") if isinstance(data2, dict) else None

            # 3) Normalize to expected provider skill shape
            out: List[Dict[str, Any]] = []
            if isinstance(rows, list):
                try:
                    rows_sorted = sorted(rows, key=lambda r: ((r or {}).get("order") or 0))
                except Exception:
                    rows_sorted = rows
                for r in rows_sorted:
                    if not isinstance(r, dict):
                        continue
                    skill = r.get("skill") if isinstance(r.get("skill"), dict) else {}
                    sid_raw = (skill.get("id") if isinstance(skill, dict) else None) or r.get("skillId")
                    sid = str(sid_raw).strip() if sid_raw is not None else ""
                    if not sid:
                        continue
                    title = None
                    if isinstance(skill, dict):
                        title = skill.get("title") or skill.get("name")
                    name = str(title or sid)
                    category = (skill.get("category") if isinstance(skill, dict) else None)
                    out.append({"id": sid, "name": name, "category": category})
            if out:
                return out
            return _fallback_from_env()
    except Exception:
        return _fallback_from_env()


# ===== Convex helpers for transcripts & session state =====
def _convex_base_url() -> Optional[str]:
    try:
        base = (os.getenv("CONVEX_URL") or "").strip()
        return base or None
    except Exception:
        return None


def _convex_timeout_seconds() -> float:
    try:
        return float(os.getenv("CONVEX_TIMEOUT_SECONDS") or os.getenv("AI_HTTP_TIMEOUT_SECONDS") or 10)
    except Exception:
        return 10.0


async def _convex_query(path: str, args: Dict[str, Any], *, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    base = _convex_base_url()
    if not base:
        return None
    url = base.rstrip("/") + "/api/query"
    payload = {"path": path, "args": args, "format": "json"}
    t = _convex_timeout_seconds() if timeout is None else timeout
    async with httpx.AsyncClient(timeout=t) as client:
        resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
        if resp.status_code >= 400:
            return None
        try:
            return resp.json()
        except Exception:
            return None


async def _convex_mutation(path: str, args: Dict[str, Any], *, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    base = _convex_base_url()
    if not base:
        return None
    url = base.rstrip("/") + "/api/mutation"
    payload = {"path": path, "args": args, "format": "json"}
    t = _convex_timeout_seconds() if timeout is None else timeout
    async with httpx.AsyncClient(timeout=t) as client:
        resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
        if resp.status_code >= 400:
            return None
        try:
            return resp.json()
        except Exception:
            return None


async def _get_session_doc(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        data = await _convex_query("functions/sessions:getBySessionId", {"sessionId": session_id})
        if isinstance(data, dict):
            # { status: 'success', value: {...} } shape from Convex HTTP
            val = data.get("value") if data.get("status") != "error" else None
            return val if isinstance(val, dict) else None
    except Exception:
        return None
    return None


async def _get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch user profile from Convex (displayName, email, avatarUrl, bio)."""
    try:
        data = await _convex_query("functions/users:getProfile", {"userId": user_id})
        if isinstance(data, dict):
            # Convex HTTP shape: { status, value }
            val = data.get("value") if data.get("status") != "error" else None
            return val if isinstance(val, dict) else None
    except Exception:
        return None
    return None


async def _list_user_goals(user_id: str) -> List[Dict[str, Any]]:
    """Fetch user goals from Convex; returns a list of goals (title, status, etc.)."""
    try:
        data = await _convex_query("functions/users:listGoals", {"userId": user_id})
        if isinstance(data, dict):
            val = data.get("value") if data.get("status") != "error" else None
            return val if isinstance(val, list) else []
    except Exception:
        return []
    return []


def _sha256_hex(s: str) -> str:
    try:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _chat_context_limit_default() -> int:
    """Default number of recent messages to include in chat context.

    Reads CHAT_CONTEXT_LIMIT from env, clamps to [1, 200], defaults to 10.
    """
    try:
        v = int((os.getenv("CHAT_CONTEXT_LIMIT") or "10").strip())
    except Exception:
        v = 10
    if v < 1:
        v = 1
    if v > 200:
        v = 200
    return v


def _chat_item_word_cap_default() -> int:
    """Optional per-message word cap. 0 disables word-based item cap."""
    try:
        return int((os.getenv("CHAT_CONTEXT_ITEM_WORD_CAP") or "0").strip())
    except Exception:
        return 0


def _chat_total_word_cap_default() -> int:
    """Optional total context word cap. 0 disables word-based total cap."""
    try:
        return int((os.getenv("CHAT_CONTEXT_TOTAL_WORD_CAP") or "0").strip())
    except Exception:
        return 0


def _chat_item_char_cap_default() -> int:
    """Per-message character cap (fallback when word cap disabled)."""
    try:
        return int((os.getenv("CHAT_CONTEXT_ITEM_CHAR_CAP") or "240").strip())
    except Exception:
        return 240


def _chat_total_char_cap_default() -> int:
    """Total context character cap (fallback when word cap disabled)."""
    try:
        return int((os.getenv("CHAT_CONTEXT_TOTAL_CHAR_CAP") or "6000").strip())
    except Exception:
        return 6000


def _truncate_by_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    try:
        words = re.findall(r"\S+", text)
        if len(words) <= max_words:
            return text
        truncated = " ".join(words[:max_words])
        return (truncated + "...") if truncated else text
    except Exception:
        return text


async def _should_persist_transcripts(session_id: Optional[str]) -> bool:
    """Determine if transcripts should be persisted.

    Precedence:
      1) Global env toggle PERSIST_TRANSCRIPTS_ENABLED (default: 0/disabled)
      2) If Convex is configured, check session.state.transcriptsPersistOptOut
    """
    enabled = os.getenv("PERSIST_TRANSCRIPTS_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        try:
            TRANSCRIPT_PERSIST_TOTAL.labels(outcome="disabled", role="unknown").inc()
        except Exception:
            pass
        return False
    if not session_id:
        try:
            TRANSCRIPT_PERSIST_TOTAL.labels(outcome="no_session", role="unknown").inc()
        except Exception:
            pass
        return False
    # If Convex absent, we cannot check opt-out but still respect global toggle
    if not _convex_base_url():
        return enabled
    try:
        doc = await _get_session_doc(session_id)
        state = (doc or {}).get("state") if isinstance(doc, dict) else None
        if isinstance(state, dict) and bool(state.get("transcriptsPersistOptOut")):
            try:
                TRANSCRIPT_PERSIST_TOTAL.labels(outcome="optout", role="unknown").inc()
            except Exception:
                pass
            return False
    except Exception:
        pass
    return True


def _classifier_context_limit_default() -> int:
    """Default number of recent messages to include in classifier context.

    Reads CLASSIFIER_CONTEXT_LIMIT from env, clamps to [1, 50], defaults to 6.
    """
    try:
        v = int((os.getenv("CLASSIFIER_CONTEXT_LIMIT") or "6").strip())
    except Exception:
        v = 6
    if v < 1:
        v = 1
    if v > 50:
        v = 50
    return v


async def _build_classifier_context_summary(session_id: Optional[str]) -> str:
    """Build a brief plain-text context summary for the classifier.

    Uses recent transcript events for the session (ignoring group boundaries) to
    keep the interface simple and provider-agnostic.
    """
    try:
        limit = _classifier_context_limit_default()
        events = await _fetch_transcript_events_for_context(session_id, group_id=None, limit=limit)
        # Render as compact lines: "u: text" / "a: text"
        lines: List[str] = []
        for ev in events:
            role = (ev.get("role") or "").lower()
            tag = "u" if role == "user" else ("a" if role == "assistant" else "?")
            txt = (ev.get("content") or "").replace("\n", " ").strip()
            if len(txt) > 240:
                txt = txt[:237] + "..."
            lines.append(f"{tag}: {txt}")
        if not lines:
            try:
                CLASSIFIER_CONTEXT_BUILD_TOTAL.labels(outcome="empty").inc()
            except Exception:
                pass
            return ""
        context = "; ".join(lines)
        # Hard cap to avoid overly long prompts to providers
        if len(context) > 2000:
            context = context[:1997] + "..."
        # Debug: log built classifier context summary
        try:
            logger.info(
                json.dumps(
                    {
                        "event": "classifier_context_built",
                        "sessionId": session_id,
                        "limit": limit,
                        "lines": lines,
                        "context": context,
                        "context_len": len(context),
                    }
                )
            )
        except Exception:
            pass
        # Metrics for context summary
        try:
            CLASSIFIER_CONTEXT_BUILD_TOTAL.labels(outcome="success").inc()
            CLASSIFIER_CONTEXT_MESSAGES.observe(len(lines))
            CLASSIFIER_CONTEXT_LENGTH_CHARS.observe(len(context))
        except Exception:
            pass
        return context
    except Exception:
        try:
            CLASSIFIER_CONTEXT_BUILD_TOTAL.labels(outcome="error").inc()
        except Exception:
            pass
        return ""


async def _persist_interaction_if_configured(
    session_id: Optional[str],
    group_id: Optional[str],
    message_id: Optional[str],
    role: Optional[str],
    content: Optional[str],
    ts_ms: Optional[int],
) -> None:
    try:
        if not await _should_persist_transcripts(session_id):
            return
        # Allow session-only persistence: group_id is optional
        if not (session_id and message_id and role and ts_ms and ts_ms > 0):
            try:
                TRANSCRIPT_PERSIST_TOTAL.labels(outcome="invalid_args", role=str(role or "unknown")).inc()
            except Exception:
                pass
            return
        text = (content or "")
        # contentHash required by Convex schema
        ch = _sha256_hex(f"{role}|{text}")
        args = {
            "sessionId": session_id,
            "messageId": message_id,
            "role": "assistant" if role == "assistant" else "user",
            "contentHash": ch,
            "text": text,
            "audioUrl": None,
            "ts": int(ts_ms),
        }
        # Only include groupId when provided (Convex validator expects optional/undefined, not null)
        if group_id:
            args["groupId"] = group_id
        t0 = time.perf_counter()
        try:
            await _convex_mutation("functions/interactions:appendInteraction", args)
            try:
                TRANSCRIPT_PERSIST_TOTAL.labels(outcome="success", role=str(role or "unknown")).inc()
            except Exception:
                pass
        except Exception:
            try:
                TRANSCRIPT_PERSIST_TOTAL.labels(outcome="error", role=str(role or "unknown")).inc()
            except Exception:
                pass
        finally:
            try:
                TRANSCRIPT_PERSIST_SECONDS.observe(time.perf_counter() - t0)
            except Exception:
                pass
    except Exception:
        # Best-effort; never fail caller
        pass


async def _fetch_transcript_events_for_context(
    session_id: Optional[str],
    group_id: Optional[str],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch recent transcript events to include in chat context.

    If CHAT_CONTEXT_FROM_CONVEX=1 and Convex is configured, fetch via interactions queries.
    Otherwise, fall back to in-memory transcripts and optional span slicing when group_id provided.
    """
    # Resolve default limit from environment if not provided, and clamp
    if not isinstance(limit, int) or limit <= 0:
        limit = _chat_context_limit_default()
    else:
        limit = max(1, min(200, limit))
    out: List[Dict[str, Any]] = []
    use_convex = os.getenv("CHAT_CONTEXT_FROM_CONVEX", "0").strip().lower() in ("1", "true", "yes", "on")
    if use_convex and _convex_base_url() and session_id:
        try:
            if group_id:
                data = await _convex_query("functions/interactions:listByGroup", {"groupId": group_id, "limit": max(1, min(200, limit))})
            else:
                data = await _convex_query("functions/interactions:listBySession", {"sessionId": session_id, "limit": max(1, min(200, limit))})
            docs = (data or {}).get("value") if isinstance(data, dict) else None
            if isinstance(docs, list):
                for d in docs[-limit:]:
                    if isinstance(d, dict):
                        out.append({
                            "role": d.get("role"),
                            "content": d.get("text") or "",
                            "ts": d.get("ts") or d.get("createdAt"),
                        })
                return out
        except Exception:
            out = []
    # Fallback to in-memory
    try:
        transcripts: Dict[str, List[Dict[str, Any]]] = app.state.session_transcripts  # type: ignore[attr-defined]
        events = transcripts.get(session_id or "", []) or []
        if group_id:
            spans: Dict[Tuple[str, str], Dict[str, int]] = app.state.group_spans  # type: ignore[attr-defined]
            span = spans.get((session_id or "", group_id), {}) or {}
            s = span.get("start_index", 0)
            e = span.get("end_index", len(events))
            if s < 0:
                s = 0
            if e is None or e > len(events):
                e = len(events)
            if e < s:
                e = s
            events = events[s:e]
        # Keep last N
        events = events[-limit:]
        out = [{"role": ev.get("role"), "content": ev.get("content") or "", "ts": ev.get("ts")} for ev in events]
    except Exception:
        out = []
    return out


def _aggregate_scores(skill_results: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, float]]:
    """Aggregate categories and scores across per-skill results via simple average.

    Returns (categories, scores).
    """
    if not skill_results:
        return RUBRIC_V1_CATEGORIES, {}
    # Union categories
    cat_set: Set[str] = set()
    for r in skill_results:
        for c in (r.get("categories") or []):
            if isinstance(c, str) and c:
                cat_set.add(c)
    categories = sorted(cat_set) if cat_set else RUBRIC_V1_CATEGORIES
    # Average scores where present
    sums: Dict[str, float] = {c: 0.0 for c in categories}
    counts: Dict[str, int] = {c: 0 for c in categories}
    for r in skill_results:
        sc = r.get("scores") or {}
        if isinstance(sc, dict):
            for k, v in sc.items():
                if k in sums:
                    try:
                        x = float(v)
                    except Exception:
                        continue
                    sums[k] += x
                    counts[k] += 1
    agg: Dict[str, float] = {}
    for c in categories:
        n = counts.get(c, 0)
        if n > 0:
            agg[c] = round(sums[c] / n, 3)
    return categories, agg

async def _run_assessment_job(
    session_id: str,
    group_id: str,
    request_id: Optional[str],
    rubric_version: str = "v2",
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

        # If assess provider is enabled, attempt provider-based assessment first.
        # Fallback to deterministic stub if provider is disabled or fails.
        provider_succeeded = False
        use_provider = os.getenv("AI_ASSESS_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
        if use_provider:
            try:
                client = get_assess_client()
                provider_name = getattr(client, "provider_name", "unknown")
                model_name = getattr(client, "model", "unknown")
                # Build minimal transcript for provider [ {role, content}, ... ]
                transcript = [
                    {"role": (e.get("role") or ""), "content": (e.get("content") or "")}
                    for e in slice_events
                ]
                # Fetch tracked skills for this session
                skills = await _get_tracked_skills_for_session(session_id)
                if skills:
                    # Fan out concurrent assessment calls per skill with per-call and group timeouts
                    async def _call_one(skill_obj: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str], float]:
                        t1 = time.perf_counter()
                        try:
                            res = await asyncio.wait_for(
                                client.assess(transcript, rubric_version=rubric_version, request_id=request_id, skill=skill_obj),
                                timeout=max(0.001, ASSESS_PER_SKILL_TIMEOUT_MS / 1000.0),
                            )
                            dt = time.perf_counter() - t1
                            return res or {}, None, dt
                        except asyncio.TimeoutError:
                            dt = time.perf_counter() - t1
                            return {}, "timeout", dt
                        except Exception as e:  # pragma: no cover - provider error handling
                            dt = time.perf_counter() - t1
                            return {}, str(e), dt

                    # Create tasks so we can enforce a group deadline
                    task_map: Dict[asyncio.Task, Tuple[int, Dict[str, Any]]] = {}
                    for idx, s in enumerate(skills):
                        t = asyncio.create_task(_call_one(s))
                        task_map[t] = (idx, s)
                    done, pending = await asyncio.wait(task_map.keys(), timeout=max(0.001, ASSESS_GROUP_TIMEOUT_MS / 1000.0))
                    # Collect results in original skill order
                    results_list: List[Tuple[Dict[str, Any], Optional[str], float]] = [({}, "group_timeout", 0.0)] * len(skills)
                    for t in done:
                        idx, _s = task_map[t]
                        try:
                            results_list[idx] = t.result()
                        except Exception as e:
                            results_list[idx] = ({}, str(e), 0.0)
                    for t in pending:
                        # Cancel pending tasks and mark as group timeout
                        try:
                            t.cancel()
                        except Exception:
                            pass
                    skill_assessments: List[Dict[str, Any]] = []
                    skill_errors: List[Dict[str, Any]] = []
                    ok_results: List[Dict[str, Any]] = []
                    for s, (res, err, dt) in zip(skills, results_list):
                        sid = str(s.get("id") or "").strip()
                        sid_hash = _skill_hash(sid)
                        if err is not None:
                            # Metrics and logs for error
                            try:
                                reason = "timeout" if err in ("timeout", "group_timeout") else "exception"
                                ASSESS_SKILL_ERRORS_TOTAL.labels(provider=provider_name, model=model_name, reason=reason).inc()
                            except Exception:
                                pass
                            try:
                                logger.info(json.dumps({
                                    "event": "assessment_skill_error",
                                    "requestId": request_id,
                                    "trackedSkillIdHash": tracked_hash,
                                    "skillId": sid or None,
                                    "skillHash": sid_hash,
                                    "sessionId": session_id,
                                    "groupId": group_id,
                                    "error": err,
                                }))
                            except Exception:
                                pass
                            # Capture per-skill error for group summary
                            skill_errors.append({
                                "skill": {k: s.get(k) for k in ("id", "name", "category") if k in s},
                                "error": err,
                            })
                            continue
                        # Success
                        try:
                            ASSESS_SKILL_SECONDS.labels(provider=provider_name, model=model_name, rubric_version=rubric_version).observe(dt)
                        except Exception:
                            pass
                        try:
                            logger.info(json.dumps({
                                "event": "assessment_skill_complete",
                                "requestId": request_id,
                                "trackedSkillIdHash": tracked_hash,
                                "skillId": sid or None,
                                "skillHash": sid_hash,
                                "sessionId": session_id,
                                "groupId": group_id,
                                "latency_ms": int(dt * 1000),
                            }))
                        except Exception:
                            pass
                        skill_assessments.append({
                            "skill": {k: s.get(k) for k in ("id", "name", "category") if k in s},
                            "result": res,
                        })
                        ok_results.append(res)

                    if ok_results:
                        cats, agg_scores = _aggregate_scores(ok_results)
                        scores = agg_scores
                        if return_summary:
                            # Build an overall summary with per-skill details attached
                            highlights: List[str] = []
                            recommendations: List[str] = []
                            for r in ok_results:
                                for h in (r.get("highlights") or []):
                                    if isinstance(h, str) and h.strip():
                                        highlights.append(h.strip())
                                for rec in (r.get("recommendations") or []):
                                    if isinstance(rec, str) and rec.strip():
                                        recommendations.append(rec.strip())
                            # Stable ordering by skill id
                            try:
                                skill_assessments.sort(key=lambda it: str(((it or {}).get("skill") or {}).get("id") or ""))
                                skill_errors.sort(key=lambda it: str(((it or {}).get("skill") or {}).get("id") or ""))
                            except Exception:
                                pass
                            # Aggregate optional tokens/cost/latency
                            tokens_in = 0
                            tokens_out = 0
                            cost_usd = 0.0
                            latency_ms_total = 0
                            for item in skill_assessments:
                                m = ((item.get("result") or {}).get("meta") or {}) if isinstance(item.get("result"), dict) else {}
                                try:
                                    ti = int(m.get("tokensIn") or m.get("promptTokens") or 0)
                                except Exception:
                                    ti = 0
                                try:
                                    to = int(m.get("tokensOut") or m.get("completionTokens") or 0)
                                except Exception:
                                    to = 0
                                try:
                                    c = float(m.get("costUsd") or 0.0)
                                except Exception:
                                    c = 0.0
                                try:
                                    lm = int(m.get("latencyMs") or 0)
                                except Exception:
                                    lm = 0
                                tokens_in += max(0, ti)
                                tokens_out += max(0, to)
                                cost_usd += max(0.0, c)
                                latency_ms_total += max(0, lm)
                            summary_obj = {
                                "rubricVersion": rubric_version,
                                "categories": cats,
                                "scores": scores,
                                "highlights": (highlights[:3] or ["placeholder"]),
                                "recommendations": (recommendations[:3] or ["placeholder"]),
                                "meta": {
                                    "provider": provider_name,
                                    "model": model_name,
                                    "skillsCount": len(skill_assessments),
                                    "tokensIn": tokens_in or None,
                                    "tokensOut": tokens_out or None,
                                    "costUsd": round(cost_usd, 6) if cost_usd else None,
                                    "latencyMsTotal": latency_ms_total or None,
                                },
                                "skillAssessments": skill_assessments,
                                "errors": skill_errors,
                            }
                            # Emit aggregated token/cost metrics for observability
                            try:
                                if tokens_in:
                                    ASSESS_TOKENS_TOTAL.labels(
                                        direction="in",
                                        provider=provider_name,
                                        model=model_name,
                                        rubric_version=rubric_version,
                                    ).inc(tokens_in)
                                if tokens_out:
                                    ASSESS_TOKENS_TOTAL.labels(
                                        direction="out",
                                        provider=provider_name,
                                        model=model_name,
                                        rubric_version=rubric_version,
                                    ).inc(tokens_out)
                                if cost_usd:
                                    ASSESS_COST_USD_TOTAL.labels(
                                        provider=provider_name,
                                        model=model_name,
                                        rubric_version=rubric_version,
                                    ).inc(cost_usd)
                            except Exception:
                                pass
                            provider_succeeded = True
                else:
                    # No skills tracked -> single provider call (back-compat)
                    result = await client.assess(transcript, rubric_version=rubric_version, request_id=request_id)
                    prov_scores = dict((result or {}).get("scores") or {})
                    if prov_scores:
                        scores = {k: float(v) for k, v in prov_scores.items() if isinstance(v, (int, float, str))}
                        if return_summary:
                            summary_obj = result or {}
                            # Emit token/cost metrics when provided by provider meta
                            try:
                                m = dict((summary_obj.get("meta") or {}))
                                ti = int(m.get("tokensIn") or m.get("promptTokens") or 0)
                                to = int(m.get("tokensOut") or m.get("completionTokens") or 0)
                                c = float(m.get("costUsd") or 0.0)
                                if ti:
                                    ASSESS_TOKENS_TOTAL.labels(
                                        direction="in",
                                        provider=provider_name,
                                        model=model_name,
                                        rubric_version=rubric_version,
                                    ).inc(ti)
                                if to:
                                    ASSESS_TOKENS_TOTAL.labels(
                                        direction="out",
                                        provider=provider_name,
                                        model=model_name,
                                        rubric_version=rubric_version,
                                    ).inc(to)
                                if c:
                                    ASSESS_COST_USD_TOTAL.labels(
                                        provider=provider_name,
                                        model=model_name,
                                        rubric_version=rubric_version,
                                    ).inc(c)
                            except Exception:
                                pass
                        provider_succeeded = True
            except Exception:
                # Any provider error -> fall back to deterministic path
                provider_succeeded = False

        # If scores are empty at this point, record a metric and leave them empty for the caller
        if not scores:
            try:
                prov = provider_name or "unknown"
                mdl = model_name or "unknown"
                reason = "provider_empty" if provider_succeeded else "provider_error"
                ASSESS_EMPTY_SCORES_TOTAL.labels(provider=prov, model=mdl, rubric_version=rubric_version, reason=reason).inc()
                logger.info(json.dumps({
                    "event": "assessments_empty_scores",
                    "requestId": request_id,
                    "trackedSkillIdHash": tracked_hash,
                    "sessionId": session_id,
                    "groupId": group_id,
                    "provider": prov,
                    "model": mdl,
                    "rubricVersion": rubric_version,
                    "reason": reason,
                }))
            except Exception:
                pass
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
                        summary_obj = _compute_rubric_v1_summary_from_spans(session_id, group_id, scores, rubric_version="v2")
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
                        v2_payload = _build_v2_persist_payload(summary_obj or {}, scores)
                        await _persist_assessment_if_configured(session_id, group_id, v2_payload, "v2", request_id)
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
# Timeouts
ASSESS_PER_SKILL_TIMEOUT_MS = _env_int("ASSESS_PER_SKILL_TIMEOUT_MS", 8000)
ASSESS_GROUP_TIMEOUT_MS = _env_int("ASSESS_GROUP_TIMEOUT_MS", 15000)

# Network-specific timeouts for better resilience
AI_HTTP_TIMEOUT_SECONDS = _env_int("AI_HTTP_TIMEOUT_SECONDS", 30)
AI_HTTP_CONNECT_TIMEOUT_SECONDS = _env_int("AI_HTTP_CONNECT_TIMEOUT_SECONDS", 10)
AI_HTTP_READ_TIMEOUT_SECONDS = _env_int("AI_HTTP_READ_TIMEOUT_SECONDS", 30)
AI_CHAT_TTFT_TIMEOUT_SECONDS = _env_int("AI_CHAT_TTFT_TIMEOUT_SECONDS", 8)  # Increased from 5
AI_CHAT_PROMPT_TIMEOUT_SECONDS = _env_int("AI_CHAT_PROMPT_TIMEOUT_SECONDS", 3)

# Interaction/session safety guards
# - Max turns per multi-turn interaction before we force an 'end'
# - Idle timeout (reserved for future use; enforced by higher layers or future PR)
INTERACTION_MAX_TURNS = _env_int("INTERACTION_MAX_TURNS", 12)
INTERACTION_IDLE_TIMEOUT_MS = _env_int("INTERACTION_IDLE_TIMEOUT_MS", 300000)
INTERACTION_GUARDS_ENABLED = _env_bool("INTERACTION_GUARDS_ENABLED", False)
STRICT_START_GUARD = _env_bool("STRICT_START_GUARD", False)
# During pytest, force guards off for deterministic behavior
if "PYTEST_CURRENT_TEST" in os.environ:
    INTERACTION_GUARDS_ENABLED = False

# Hashing salt for skill IDs
SKILL_HASH_SALT = os.getenv("SKILL_HASH_SALT", "").strip()

def _skill_hash(skill_id: Optional[str]) -> Optional[str]:
    try:
        sid = (skill_id or "").strip()
        if not sid:
            return None
        data = (SKILL_HASH_SALT + sid).encode("utf-8") if SKILL_HASH_SALT else sid.encode("utf-8")
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return None

 

# Queue provider (Phase 2) — SQS feature flag and configuration
USE_SQS = _env_bool("USE_SQS", False)
AWS_REGION = os.getenv("AWS_REGION", "").strip()
AWS_SQS_QUEUE_URL = os.getenv("AWS_SQS_QUEUE_URL", "").strip()
AWS_ENDPOINT_URL_SQS = os.getenv("AWS_ENDPOINT_URL_SQS", "").strip()

# Optional persistence callback (HTTP)
PERSIST_ASSESSMENTS_URL = os.getenv("PERSIST_ASSESSMENTS_URL", "").strip()
PERSIST_ASSESSMENTS_SECRET = os.getenv("PERSIST_ASSESSMENTS_SECRET", "").strip()

# Session summary env configuration
SUMMARY_ENABLED = _env_bool("SUMMARY_ENABLED", True)
SUMMARY_WORKER_CONCURRENCY = _env_int("SUMMARY_WORKER_CONCURRENCY", 1)
SUMMARY_WINDOW_N = _env_int("SUMMARY_WINDOW_N", 10)
SUMMARY_PERSIST_URL = os.getenv("SUMMARY_PERSIST_URL", "").strip()
SUMMARY_PERSIST_AUTH = os.getenv("SUMMARY_PERSIST_AUTH", "").strip()

def _as_str_list(val: Any) -> List[str]:
    out: List[str] = []
    if isinstance(val, list):
        for x in val:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
    elif isinstance(val, str) and val.strip():
        out.append(val.strip())
    return out

def _build_v2_persist_payload(summary: Dict[str, Any], scores: Dict[str, float]) -> Dict[str, Any]:
    """Map current summary/result structure to the new skill-aligned v2 shape.

    v2 minimal shape:
      {
        "skillAssessments": [
          {"skillHash": str, "level": float?, "metCriteria": [], "unmetCriteria": [], "feedback": []}
        ],
        "meta": { provider, model, skillsCount }
      }
    """
    skills_raw = summary.get("skillAssessments") or []
    v2_items: List[Dict[str, Any]] = []
    for it in skills_raw:
        skill = (it or {}).get("skill") or {}
        res = (it or {}).get("result") or {}
        sid = str(skill.get("id") or "").strip()
        v2_items.append({
            "skillHash": _skill_hash(sid),
            "level": res.get("level"),
            "metCriteria": list(res.get("metCriteria") or []),
            "unmetCriteria": list(res.get("unmetCriteria") or []),
            "feedback": _as_str_list(res.get("feedback") or res.get("recommendations") or res.get("highlights") or []),
        })
    meta = summary.get("meta") or {}
    return {
        "skillAssessments": v2_items,
        "meta": {
            "provider": meta.get("provider"),
            "model": meta.get("model"),
            "skillsCount": meta.get("skillsCount"),
        },
    }

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


# -----------------------------
# Session Summary: helper and worker
# -----------------------------

def _build_rolling_summary_text(messages: List[Dict[str, Any]], *, max_chars: int = 2000) -> str:
    # Deterministic compact digest of last N messages: "role: content" lines, whitespace collapsed
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "").strip()[:12]
        content = re.sub(r"\s+", " ", (m.get("content") or "").strip())
        if len(content) > 220:
            content = content[:217] + "…"
        parts.append(f"{role}: {content}")
    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[-max_chars:]
    return out


async def _persist_summary_if_configured(payload: Dict[str, Any], *, request_id: Optional[str] = None) -> None:
    if not SUMMARY_PERSIST_URL:
        return
    headers = {"content-type": "application/json"}
    if SUMMARY_PERSIST_AUTH:
        headers["authorization"] = SUMMARY_PERSIST_AUTH
    if request_id:
        headers["x-request-id"] = str(request_id)

    def _do_request():
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(SUMMARY_PERSIST_URL, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                try:
                    resp.read()
                except Exception:
                    pass
        except Exception as e:
            try:
                logger.warning(json.dumps({
                    "event": "summary_persist_error",
                    "sessionId": payload.get("sessionId"),
                    "error": str(e),
                }))
            except Exception:
                pass

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _do_request)


async def _summary_worker(app: FastAPI, worker_index: int = 0):
    """Background worker for session summary jobs.

    Job item: (session_id: str, version: int, enq_ts: float)
    """
    queue: asyncio.Queue[Tuple[str, int, float]] = app.state.summary_queue
    while True:
        try:
            session_id, version, enq_ts = await queue.get()
            try:
                SUMMARY_QUEUE_DEPTH.set(queue.qsize())
            except Exception:
                pass
            t0 = time.perf_counter()
            # Build summary from last N messages
            try:
                transcripts: Dict[str, List[Dict[str, Any]]] = app.state.session_transcripts
                all_msgs = transcripts.get(session_id, [])
                window = int(SUMMARY_WINDOW_N) if SUMMARY_WINDOW_N > 0 else 10
                msgs = all_msgs[-window:]
            except Exception:
                msgs = []
            summary_text = _build_rolling_summary_text(msgs)
            now_ms = int(time.time() * 1000)
            row = {"version": int(version), "updatedAt": now_ms, "summaryText": summary_text}
            # Store in memory
            try:
                store: Dict[str, Dict[str, Any]] = app.state.session_summaries
                store[session_id] = row
            except Exception:
                pass

            # Enqueue latency metric
            try:
                if enq_ts:
                    SUMMARY_ENQUEUE_LAT_SECONDS.observe(max(0.0, time.perf_counter() - enq_ts))
            except Exception:
                pass

            # Optional persistence (write-behind)
            try:
                payload = {"sessionId": session_id, **row}
                await _persist_summary_if_configured(payload)
            except Exception:
                pass

            # Metrics: success and duration
            try:
                SUMMARY_JOBS_TOTAL.labels(status="ok").inc()
                SUMMARY_JOB_SECONDS.observe(time.perf_counter() - t0)
            except Exception:
                pass
        except asyncio.CancelledError:
            break
        except Exception:
            try:
                SUMMARY_JOBS_TOTAL.labels(status="error").inc()
            except Exception:
                pass
            await asyncio.sleep(0.25)


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
            tracked_hash = _skill_hash(_tsid) if _tsid else None
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
    ts_ms: int
    try:
        transcripts: Dict[str, List[Dict[str, Any]]] = app.state.session_transcripts  # type: ignore[attr-defined]
        lst = transcripts.setdefault(session_id, [])
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
        # Strict every-N summarization enqueue (independent of assessments)
        if SUMMARY_ENABLED and SUMMARY_WINDOW_N > 0:
            try:
                count = len(lst)
                if count % SUMMARY_WINDOW_N == 0:
                    version = count // SUMMARY_WINDOW_N
                    q: asyncio.Queue = app.state.summary_queue  # type: ignore[attr-defined]
                    enq_ts = time.perf_counter()
                    await q.put((session_id, version, enq_ts))
                    try:
                        SUMMARY_QUEUE_DEPTH.set(q.qsize())
                    except Exception:
                        pass
                    try:
                        app.state.summary_enqueued_ts[session_id] = enq_ts  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        logger.info(json.dumps({
                            "event": "summary_enqueued",
                            "sessionId": session_id,
                            "version": version,
                            "count": count,
                            "window": SUMMARY_WINDOW_N,
                        }))
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        # Do not fail ingestion if transcript buffering has issues
        evt_idx = -1  # sentinel
        try:
            ts_ms = int(ts) if ts is not None else int(time.time() * 1000)
        except Exception:
            ts_ms = int(time.time() * 1000)

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
            # Build compact context and embed into the content for provider-agnostic interfaces
            try:
                ctx = await _build_classifier_context_summary(session_id)
            except Exception:
                ctx = ""
            infused = (f"ctx: {ctx}\nmsg: {content or ''}" if ctx else (content or ""))
            # Debug: log exact classifier request payload
            try:
                logger.info(
                    json.dumps(
                        {
                            "event": "classifier_request",
                            "requestId": request_id,
                            "trackedSkillIdHash": tracked_hash,
                            "sessionId": session_id,
                            "messageId": message_id,
                            "provider": cls_provider,
                            "model": cls_model,
                            "role": role,
                            "turnCount": turn_count,
                            "ctx_len": len(ctx or ""),
                            "infused": infused,
                        }
                    )
                )
            except Exception:
                pass
            cls = await client.classify(role, infused, turn_count, request_id=request_id)
        except Exception:
            # Fallback to deterministic mock classifier
            try:
                client = get_classifier_client(provider="mock", model=os.getenv("AI_CLASSIFIER_MODEL", "").strip() or None)
                cls_provider = getattr(client, "provider_name", None) or "mock"
                cls_model = getattr(client, "model", None)
                try:
                    ctx = await _build_classifier_context_summary(session_id)
                except Exception:
                    ctx = ""
                infused = (f"ctx: {ctx}\nmsg: {content or ''}" if ctx else (content or ""))
                # Debug: log exact classifier request payload (fallback)
                try:
                    logger.info(
                        json.dumps(
                            {
                                "event": "classifier_request",
                                "requestId": request_id,
                                "trackedSkillIdHash": tracked_hash,
                                "sessionId": session_id,
                                "messageId": message_id,
                                "provider": cls_provider,
                                "model": cls_model,
                                "role": role,
                                "turnCount": turn_count,
                                "ctx_len": len(ctx or ""),
                                "infused": infused,
                            }
                        )
                    )
                except Exception:
                    pass
                cls = await client.classify(role, infused, turn_count, request_id=request_id)
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
    decision_from_heuristic = False

    if not accepted:
        # Apply heuristic fallback or override if low confidence or abstain decision
        heuristic = _heuristic_decision(role, content or "", bool(session_state.get("active")))
        # Use heuristic if confidence is low OR if decision is abstain/ignore (regardless of confidence)
        if confidence < CLASSIFIER_CONF_LOW or decision in ["abstain", "ignore"]:
            decision = heuristic
            decision_from_heuristic = True

    # Conservative backend guards to prevent runaway multi-turns (opt-in)
    decision_pre_guards = decision
    idle_ms = 0
    if INTERACTION_GUARDS_ENABLED:
        # 1) Optionally require accepted confidence to start a session, unless the start came from
        # a backend heuristic override (tests expect heuristics to open a session).
        if STRICT_START_GUARD and decision == "start" and not accepted and not decision_from_heuristic:
            decision = "abstain"

        # Compute idle gap since last message in this session (ms)
        try:
            last_ts_val = int(session_state.get("lastTs") or ts_ms)
        except Exception:
            last_ts_val = ts_ms
        idle_ms = max(0, int(ts_ms) - int(last_ts_val))

        # 2) If classifier says continue but session is not active, promote to start
        # so the first message can open a session rather than being dropped.
        if decision == "continue" and not session_state.get("active"):
            decision = "start"

        # 3) If continue while idle beyond threshold, force an end to close the prior interaction
        if decision == "continue" and session_state.get("active") and idle_ms > INTERACTION_IDLE_TIMEOUT_MS:
            decision = "end"

        # 4) Enforce max turns: if continuing would exceed cap, end the interaction on this message
        if decision == "continue":
            projected_turns = int(session_state.get("turnCount", 0)) + 1
            if projected_turns >= INTERACTION_MAX_TURNS:
                decision = "end"

    # Telemetry polish: if a session is already active, normalize a 'start' to 'continue'
    # so analytics don't double-count starts. This does not change state behavior.
    if session_state.get("active") and decision == "start":
        decision = "continue"

    # Emit a single structured log capturing classifier outcome and any guard overrides
    try:
        logger.info(json.dumps({
            "event": "classifier_decision",
            "requestId": request_id,
            "trackedSkillIdHash": tracked_hash,
            "sessionId": session_id,
            "messageId": message_id,
            "provider": cls_provider,
            "model": cls_model,
            "role": role,
            "turnCount": turn_count,
            "activeBefore": bool(session_state.get("active")),
            "confidence": confidence,
            "accepted": accepted,
            "low": low,
            "decisionPre": decision_pre_guards,
            "decisionPost": decision,
            "heuristic": decision_from_heuristic,
            "guardsEnabled": INTERACTION_GUARDS_ENABLED,
            "strictStartGuard": STRICT_START_GUARD,
            "idleMs": idle_ms,
            "maxTurns": INTERACTION_MAX_TURNS,
        }))
    except Exception:
        pass

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
                    "rubricVersion": "v2",
                    "skillAssessments": [],
                    "meta": {"provider": "stub", "model": "stub", "skillsCount": 0},
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
                    "rubricVersion": "v2",
                    "skillAssessments": [],
                    "meta": {"provider": "stub", "model": "stub", "skillsCount": 0},
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
                            # If scores are empty, tag summary with jobError for observability
                            if not scores_sync and isinstance(summary_obj, dict) and not summary_obj.get("jobError"):
                                summary_obj = {**summary_obj, "jobError": "empty_scores_retries_exhausted"}
                            app.state.assessment_results[session_id] = {  # type: ignore[attr-defined]
                                "latestGroupId": gid_now,
                                "summary": summary_obj,
                            }
                        except Exception:
                            pass
                        # Optional HTTP persistence callback
                        try:
                            v2_payload = _build_v2_persist_payload(summary_obj or {}, scores_sync)
                            await _persist_assessment_if_configured(session_id, gid_now, v2_payload, "v2", request_id)
                        except Exception:
                            pass
                except Exception:
                    pass
        # finalize interaction
        session_state["active"] = False
    elif decision == "one_off":
        # Enqueue single-turn assessment immediately
        gid = str(uuid.uuid4())
        # placeholder removed
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
                    if not scores_sync and isinstance(summary_obj, dict) and not summary_obj.get("jobError"):
                        summary_obj = {**summary_obj, "jobError": "empty_scores_retries_exhausted"}
                    app.state.assessment_results[session_id] = {  # type: ignore[attr-defined]
                        "latestGroupId": gid,
                        "summary": summary_obj,
                    }
                except Exception:
                    pass
                # Optional HTTP persistence callback
                try:
                    v2_payload = _build_v2_persist_payload(summary_obj or {}, scores_sync)
                    await _persist_assessment_if_configured(session_id, gid, v2_payload, "v2", request_id)
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
    # Best-effort persistence to Convex (respects privacy opt-out)
    try:
        gid_for_persist: Optional[str] = None
        if decision == "one_off":
            gid_for_persist = session_state.get("groupId")
        else:
            gid_for_persist = session_state.get("groupId")
        await _persist_interaction_if_configured(
            session_id=session_id,
            group_id=gid_for_persist,
            message_id=message_id,
            role=role,
            content=content,
            ts_ms=ts_ms,
        )
    except Exception:
        pass
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
            tracked_hash = _skill_hash(_tsid) if _tsid else None
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
    # Raw skill id header (used for response-level filtering)
    try:
        tsid = request.headers.get("x-tracked-skill-id")
    except Exception:
        tsid = None
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
        # Apply response-level filtering when a specific tracked skill id is provided
        if tsid:
            try:
                summ_in = (data or {}).get("summary") or {}
                summ_out = _filter_summary_by_skill_id(summ_in, tsid)
                return {"sessionId": sessionId, **{**(data or {}), "summary": summ_out}}
            except Exception:
                pass
        return {"sessionId": sessionId, **(data or {})}
    # If not found, attempt to compute on-demand using last known groupId from session state
    try:
        ss_map: Dict[str, Dict[str, Any]] = getattr(app.state, "session_state", {})  # type: ignore[attr-defined]
        session_state: Dict[str, Any] = ss_map.get(decoded_id) or ss_map.get(sessionId)
    except Exception:
        session_state = None  # type: ignore[assignment]
    if session_state and session_state.get("groupId"):
        gid = str(session_state.get("groupId"))
        # Compute scores synchronously and persist like the worker
        scores_sync, summary_obj = await _run_assessment_job(decoded_id, gid, request_id=None, return_summary=True, tracked_hash=tracked_hash)
        try:
            if not scores_sync and isinstance(summary_obj, dict) and not summary_obj.get("jobError"):
                summary_obj = {**summary_obj, "jobError": "empty_scores_retries_exhausted"}
            app.state.assessment_results[decoded_id] = {  # type: ignore[attr-defined]
                "latestGroupId": gid,
                "summary": summary_obj,
            }
        except Exception:
            pass
        # Return filtered view per caller request, but persist full summary above
        try:
            resp_summary = _filter_summary_by_skill_id(summary_obj, tsid) if tsid else summary_obj
        except Exception:
            resp_summary = summary_obj
        return {"sessionId": sessionId, "latestGroupId": gid, "summary": resp_summary}
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
            "rubricVersion": "v2",
            "skillAssessments": [],
            "meta": {"provider": "stub", "model": "stub", "skillsCount": 0},
        },
    }


@app.get("/service-metrics", tags=["meta"], description="Lightweight service metrics for observability.")
async def service_metrics():
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
