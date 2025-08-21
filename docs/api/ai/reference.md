# AI API (FastAPI) — Reference

Overview of AI endpoints (chat streaming, assessments, summaries) served by the Python FastAPI service.

## Conventions
- Base URL: /v1
- Auth: service-to-service (Next.js → FastAPI) using JWT verification or shared secret for background jobs
- Request ID: X-Request-Id (propagated)
- Content-Type: application/json (requests); text/event-stream for streaming

## Endpoints (MVP)

### GET /chat/stream
- Summary: Stream tokens via SSE for chat.
- Headers: X-Request-Id, Accept: text/event-stream
- Query: prompt? (optional)
- Response: text/event-stream

### POST /assessments/run
- Summary: Start a multi-turn assessment job for a session; returns a groupId.
- Body: { "sessionId": string }
- Response: { "groupId": string, "status": "accepted" }

Example:

```bash
curl -s --json '{"sessionId":"s_demo"}' http://localhost:8000/assessments/run
```

### GET /assessments/{sessionId}
- Summary: Fetch latest assessment summary for a session.
- Response: { sessionId, latestGroupId, summary: { highlights[], recommendations[], categories[], scores{...}, meta{...}, rubricVersion, rubricKeyPoints[] } }

Example:

```bash
curl -s http://localhost:8000/assessments/s_demo | jq .
```

### GET /metrics
- Summary: Prometheus metrics endpoint for SQS + assessment worker instrumentation.
- Media type: text/plain; version=0.0.4

### GET /health
- Summary: healthcheck endpoint.

## OpenAPI Spec
- FastAPI serves /openapi.json automatically.
- Snapshot: curl http://localhost:8000/openapi.json > docs/api/ai/openapi.json

## Changelog
- 2025-08-21: Updated endpoints to assessments run/get; documented expanded summary payload (categories, scores, meta, rubricVersion, rubricKeyPoints).
- 2025-08-19: Initial stub added.
