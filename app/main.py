from fastapi import FastAPI, Request, Query
from fastapi.openapi.utils import get_openapi
from starlette.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import asyncio
import time
import json
import logging
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
    ]
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local dev"}
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[assignment]
