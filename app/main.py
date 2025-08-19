from fastapi import FastAPI

from app.middleware.request_id import RequestIdMiddleware


app = FastAPI(title="Coach Up AI API")
app.add_middleware(RequestIdMiddleware)


@app.get("/health")
async def health():
    return {"status": "ok"}
