import json
import time
import uuid
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Accepts or generates X-Request-Id, persists on request.state, and echoes on response.

    Also emits a minimal JSON log for each request with method, path, status, and latency_ms.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable):
        start = time.time()
        req_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = req_id

        response = await call_next(request)
        response.headers["X-Request-Id"] = req_id

        try:
            log = {
                "ts": int(time.time() * 1000),
                "level": "info",
                "requestId": req_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": int((time.time() - start) * 1000),
            }
            print(json.dumps(log), flush=True)
        except Exception:
            # Never fail the request on logging errors
            pass

        return response
