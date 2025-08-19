# AI API (FastAPI) — Reference

Overview of AI endpoints (chat streaming, assessments, summaries) served by the Python FastAPI service.

## Conventions
- Base URL: /v1
- Auth: service-to-service (Next.js → FastAPI) using JWT verification or shared secret for background jobs
- Request ID: X-Request-Id (propagated)
- Content-Type: application/json (requests); text/event-stream for streaming

## Endpoints (MVP)

### POST /v1/chat/stream
- Summary: Stream LLM tokens for chat interaction.
- Auth: required (service)
- Headers: X-Request-Id, Accept: text/event-stream
- Body: { "message": string, "sessionId": string }
- Response: text/event-stream

### POST /v1/assessments/batch
- Summary: Analyze a batch of interactions; return per-focus assessments.
- Auth: required (service)
- Body: { "interactionIds": string[], "rubricVersion": string }
- Response: { "groupId": string, "assessments": Assessment[] }

### POST /v1/summary/generate
- Summary: Generate end-of-session summary document.
- Auth: required (service)

### GET /health
- Summary: healthcheck endpoint.

## OpenAPI Spec
- FastAPI serves /openapi.json automatically.
- Snapshot: curl http://localhost:8000/openapi.json > docs/api/ai/openapi.json

## Changelog
- 2025-08-19: Initial stub added.
