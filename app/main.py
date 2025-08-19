from fastapi import FastAPI, Request, Query, Body
from fastapi.openapi.utils import get_openapi
from starlette.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import asyncio
import time
import json
import logging
import uuid
from typing import Optional

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
RUBRIC_V1_CATEGORIES = [
    "correctness",
    "clarity",
    "conciseness",
    "fluency",
]


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
    start = time.perf_counter()
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
        scores = {}
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
    finally:
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


@app.post(
    "/assessments/run",
    tags=["assessments"],
    description="Start a multi-turn assessment job for a session; returns a groupId.",
)
async def assessments_run(request: Request, sessionId: str = Body(..., embed=True, description="Session ID to assess")):
    group_id = str(uuid.uuid4())
    request_id = getattr(getattr(request, "state", object()), "request_id", None) or request.headers.get("x-request-id")
    # Fire-and-forget background job
    asyncio.create_task(_run_assessment_job(sessionId, group_id, request_id))
    return {"groupId": group_id, "status": "accepted"}


@app.get(
    "/assessments/{sessionId}",
    tags=["assessments"],
    description="Fetch latest assessment summary for a session (stub).",
)
async def assessments_get(sessionId: str):
    # Stub response for MVP wiring; replaced in later sprint work
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
