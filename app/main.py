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
from typing import Optional, Dict, Any, Tuple

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
    ]
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local dev"}
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[assignment]
