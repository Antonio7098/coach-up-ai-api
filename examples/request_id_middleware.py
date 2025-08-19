"""
Minimal Request ID middleware for FastAPI / Starlette.
- Accepts incoming X-Request-Id header (case-insensitive)
- Generates one if missing
- Sets request.state.request_id and response header
- Emits simple structured logs (JSON) with timing

Usage:
    from fastapi import FastAPI
    from examples.request_id_middleware import RequestIdMiddleware

    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


logger = logging.getLogger("request")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _new_request_id() -> str:
    return str(uuid.uuid4())


class RequestIdMiddleware(BaseHTTPMiddleware):
    HEADER = "X-Request-Id"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.time()
        # Get or create request id
        rid = request.headers.get(self.HEADER) or request.headers.get(self.HEADER.lower()) or _new_request_id()
        request.state.request_id = rid

        # Log inbound
        logger.info(
            json.dumps(
                {
                    "level": "info",
                    "message": "inbound_request",
                    "requestId": rid,
                    "method": request.method,
                    "path": request.url.path,
                }
            )
        )

        # Forward to handler
        response = await call_next(request)

        # Set response header and log outbound
        response.headers[self.HEADER] = rid
        duration_ms = int((time.time() - start) * 1000)
        logger.info(
            json.dumps(
                {
                    "level": "info",
                    "message": "outbound_response",
                    "requestId": rid,
                    "status": response.status_code,
                    "durationMs": duration_ms,
                }
            )
        )
        return response
