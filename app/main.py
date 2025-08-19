from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from starlette.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import asyncio

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
async def chat_stream():
    return StreamingResponse(_token_stream(), media_type="text/event-stream")

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
