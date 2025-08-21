# Coach Up AI API Docs

This folder contains human-friendly references and machine-readable specs for the AI API (FastAPI).

## Structure
- ai/
  - reference.md — overview, examples, and changelog
  - openapi.json — exported from FastAPI (/openapi.json)
- shared/
  - headers.md — auth, request ID, rate limits, content types
  - errors.md — error model and codes

## Conventions
- Versioning: path-based `/v1/...`
- Auth: service-to-service (JWT verification or shared secret)
- Request ID: `X-Request-Id` propagated from callers
- Streaming: `text/event-stream` for SSE endpoints
- Errors: consistent JSON envelope

## Maintaining the spec
- Source of truth: Pydantic models and FastAPI routes
- Snapshot OpenAPI locally: `curl http://localhost:8000/openapi.json > docs/api/ai/openapi.json`
- Optional: lint OpenAPI with Spectral in CI.

## E2E toggles
- SKIP_AI_CONTRACTS
  - When set to `1` or `true` (in frontend Playwright), AI contract tests are skipped and the FastAPI server is not started by the test runner.
  - Use for UI-only smoke runs.
- MOCK_CONVEX
  - Controlled by the frontend during E2E. When `1`, Next.js proxy uses an in-memory Convex mock; AI API behavior is unchanged.
  - Unset or `0` when running against a real Convex dev server.
