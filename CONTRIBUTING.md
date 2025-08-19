# Contributing to Coach Up (AI API)

Thanks for contributing! This repo hosts the Python FastAPI AI service.

## Dev setup
- Python 3.11+
- Optional: virtualenv or uv/poetry
- Copy `.env.example` to `.env` and fill values
- Start the app (example): `uvicorn app.main:app --reload --port 8000`

## Common tasks
- Snapshot OpenAPI: `make openapi-snapshot` (app must be running)
- Lint OpenAPI: `make openapi-lint`
- Build API docs (HTML): `make redoc-build`

## PR checklist (Definition of Done)
- [ ] References Sprint (SPR-###) and Features (FEAT-###) in PR description
- [ ] Tests updated/added (pytest) if code changed
- [ ] Observability updated (logs/metrics/alerts) if applicable
- [ ] Documentation updated (PRD/Technical Overview/Runbooks) if applicable
- [ ] API docs updated if endpoints changed (OpenAPI + reference)

## Request ID & Logging
- Accept and propagate `X-Request-Id` to correlate logs end-to-end.
- See `examples/request_id_middleware.py` for a minimal FastAPI middleware stub.
