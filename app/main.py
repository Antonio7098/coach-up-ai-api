from fastapi import FastAPI, Request, Query, Body
from fastapi.openapi.utils import get_openapi
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from urllib.parse import parse_qs
import asyncio
import time
import json
import logging
import uuid
from typing import Optional, Dict, Any, Tuple, Set
import os

from app.middleware.request_id import RequestIdMiddleware


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
@app.on_event("startup")
async def _startup_event():
    # Simple in-memory queue and results store for sprint scaffolding
    app.state.assessment_queue = asyncio.Queue()  # type: ignore[attr-defined]
    app.state.assessment_results = {}  # type: ignore[attr-defined]
    app.state.assessment_worker_task = asyncio.create_task(_assessments_worker(app))  # type: ignore[attr-defined]
    # In-memory session interaction state and processed ids
    # session_state[session_id] = {
    #   "active": bool,
    #   "groupId": Optional[str],
    #   "turnCount": int,
    #   "lastTs": Optional[int]
    # }
    app.state.session_state: Dict[str, Dict[str, Any]] = {}
    app.state.processed_message_ids: Set[str] = set()


@app.on_event("shutdown")
async def _shutdown_event():
    task = getattr(app.state, "assessment_worker_task", None)
    if task:
        task.cancel()
        try:
            await task
        except Exception:
            pass


async def _assessments_worker(app: FastAPI):
    """Background worker that processes assessment jobs from the queue.

    Job item: (session_id, group_id, request_id)
    """
    queue: asyncio.Queue[Tuple[str, str, Optional[str]]] = app.state.assessment_queue  # type: ignore[attr-defined]
    results: Dict[str, Dict[str, Any]] = app.state.assessment_results  # type: ignore[attr-defined]
    while True:
        try:
            session_id, group_id, request_id = await queue.get()
            scores = await _run_assessment_job(session_id, group_id, request_id)
            # Persist latest results in-memory (stub; replace with Convex in later sprint step)
            results[session_id] = {
                "latestGroupId": group_id,
                "summary": {
                    "highlights": ["placeholder"],
                    "recommendations": ["placeholder"],
                    "rubricVersion": "v1",
                    "categories": RUBRIC_V1_CATEGORIES,
                    "scores": scores,
                },
            }
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

async def _token_stream():
    # Minimal stub stream to validate SSE pipeline
    # Sends 5 tokens then a [DONE] marker
    for token in ["Hello", ", ", "world", "!", "\n"]:
        yield f"data: {token}\n\n"
        await asyncio.sleep(0.2)
    yield "data: [DONE]\n\n"

@app.get("/chat/stream", tags=["chat"], description="SSE stream of chat tokens.")
async def chat_stream(
    request: Request,
    prompt: Optional[str] = Query(None, description="Optional user prompt for logging/testing"),
):
    start = time.perf_counter()
    first_token_s: Optional[float] = None

    # Try to obtain request id from middleware or header
    request_id = getattr(getattr(request, "state", object()), "request_id", None) or request.headers.get("x-request-id")

    async def instrumented_stream():
        nonlocal first_token_s
        try:
            async for chunk in _token_stream():
                if first_token_s is None:
                    first_token_s = time.perf_counter() - start
                    try:
                        logger.info(
                            json.dumps(
                                {
                                    "event": "chat_stream_first_token",
                                    "requestId": request_id,
                                    "ttft_ms": int(first_token_s * 1000),
                                    "route": "/chat/stream",
                                    "prompt_present": bool(prompt),
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
                            "ttft_ms": int(first_token_s * 1000) if first_token_s is not None else None,
                            "total_ms": int(total_s * 1000),
                            "route": "/chat/stream",
                            "prompt_present": bool(prompt),
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

async def _run_assessment_job(session_id: str, group_id: str, request_id: Optional[str], rubric_version: str = "v1"):
    """Simulated multi-turn assessment job.

    Returns a dict of per-category scores in [0,1].
    """
    # TODO(SPR-002 backend): Implement multi-turn assessment pipeline
    # - Load recent interactions for sessionId (buffer)
    # - Score with rubric v1 and produce per-category metrics
    # - Persist baseline to Convex (createAssessmentGroup) if not present
    # - Persist per-turn/multi-turn assessments
    # - On finalize, aggregate and write summary document
    start = time.perf_counter()
    scores: Dict[str, float] = {}
    try:
        logger.info(
            json.dumps(
                {
                    "event": "assessments_job_start",
                    "requestId": request_id,
                    "sessionId": session_id,
                    "groupId": group_id,
                    "rubricVersion": rubric_version,
                }
            )
        )
        # Simulate rubric evaluation work
        await asyncio.sleep(0.2)
        # Produce deterministic-ish scores in [0,1] using hash of ids
        base = abs(hash((session_id, group_id))) % 1000
        for i, cat in enumerate(RUBRIC_V1_CATEGORIES):
            scores[cat] = round(((base + (i * 73)) % 1000) / 1000.0, 2)
        try:
            logger.info(
                json.dumps(
                    {
                        "event": "assessments_scores",
                        "requestId": request_id,
                        "sessionId": session_id,
                        "groupId": group_id,
                        "rubricVersion": rubric_version,
                        "scores": scores,
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
                    "sessionId": session_id,
                    "groupId": group_id,
                    "rubricVersion": rubric_version,
                    "total_ms": int(total_s * 1000),
                }
            )
        )
    except Exception:
        pass
    return scores


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
    processed: Set[str] = app.state.processed_message_ids  # type: ignore[attr-defined]
    if message_id in processed:
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

    # LLM-first classification
    turn_count = int(session_state.get("turnCount", 0))
    cls = await _classify_message_llm_stub(role, content or "", turn_count)
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
    elif decision == "continue":
        if not session_state.get("active"):
            _ensure_group(session_state)
        session_state["turnCount"] = int(session_state.get("turnCount", 0)) + 1
        session_state["active"] = True
    elif decision == "end":
        if not session_state.get("active"):
            # nothing to end; ignore
            pass
        else:
            # enqueue assessment for this group
            try:
                await app.state.assessment_queue.put((session_id, session_state.get("groupId"), request_id))  # type: ignore[attr-defined]
                enqueued = True
            except Exception:
                asyncio.create_task(_run_assessment_job(session_id, session_state.get("groupId"), request_id))
                enqueued = True
        # finalize interaction
        session_state["active"] = False
    elif decision == "one_off":
        # Enqueue single-turn assessment immediately
        gid = str(uuid.uuid4())
        try:
            await app.state.assessment_queue.put((session_id, gid, request_id))  # type: ignore[attr-defined]
            enqueued = True
        except Exception:
            asyncio.create_task(_run_assessment_job(session_id, gid, request_id))
            enqueued = True
        session_state["active"] = False
        session_state["groupId"] = gid
        session_state["turnCount"] = 1
    else:
        # ignore/abstain: no state change
        pass

    # Update last timestamp and processed set
    session_state["lastTs"] = ts
    processed.add(message_id)

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
                    "sessionId": session_id,
                    "messageId": message_id,
                    "decision": decision,
                    "confidence": confidence,
                    "accepted": accepted,
                    "low": low,
                    "enqueued": enqueued,
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
        await app.state.assessment_queue.put((session_id, group_id, request_id))  # type: ignore[attr-defined]
    except Exception:
        # Fallback to immediate task if queue not ready (e.g., during early boot/tests)
        asyncio.create_task(_run_assessment_job(session_id, group_id, request_id))
    return {"groupId": group_id, "status": "accepted"}


@app.get(
    "/assessments/{sessionId}",
    tags=["assessments"],
    description="Fetch latest assessment summary for a session (stub).",
)
async def assessments_get(sessionId: str):
    # TODO(SPR-002 backend): Fetch from Convex instead of in-memory stub
    # - Query latest summary by sessionId
    # - Respect auth and user scoping
    results: Dict[str, Dict[str, Any]] = getattr(app.state, "assessment_results", {})  # type: ignore[attr-defined]
    if sessionId in results:
        data = results[sessionId]
        return {"sessionId": sessionId, **data}
    # Fallback stub response
    return {
        "sessionId": sessionId,
        "latestGroupId": None,
        "summary": {
            "highlights": ["placeholder"],
            "recommendations": ["placeholder"],
            "rubricVersion": "v1",
            "categories": RUBRIC_V1_CATEGORIES,
        },
    }

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
