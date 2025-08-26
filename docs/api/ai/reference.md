# AI API (FastAPI) — Reference

Overview of AI endpoints (chat streaming, assessments, summaries) served by the Python FastAPI service.

## Conventions
- Base URL: /v1
- Auth: service-to-service (Next.js → FastAPI) using JWT verification or shared secret for background jobs
- Request ID: X-Request-Id (propagated)
- Content-Type: application/json (requests); text/event-stream for streaming
- Per-skill provider headers: see `docs/api/shared/headers.md` for `X-Tracked-Skill-Id` and `X-Tracked-Skill-Id-Hash` used during provider fan-out. Providers receive both headers when a skill is provided; only the hash is logged/persisted.
 - Persistence: assessment persistence is v2-only. Legacy v1 has been removed and the `ASSESS_OUTPUT_V2` flag no longer exists.

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
- Response: {
    sessionId,
    latestGroupId,
    summary: {
      highlights[],
      recommendations[],
      categories[],
      scores{...},
      meta{...},
      rubricVersion,
      rubricKeyPoints[],
      skillAssessments?: [
        {
          skill: { id: string, name?: string, category?: string, ... },
          result: {
            rubricVersion: string,
            categories: string[],
            scores: object,
            highlights?: string[],
            recommendations?: string[],
            meta?: object
          }
        }
      ]
    }
  }

Notes:
- When tracked skills are configured server-side, the service performs per-skill concurrent assessments (fan-out) and aggregates scores by averaging across skill results.
- If no skills are configured, the service performs a single assessment and `skillAssessments` may be omitted.

Example:

```bash
curl -s http://localhost:8000/assessments/s_demo | jq .
```

## Persistence (v2-only)

The AI API optionally persists assessment results to an external service when `PERSIST_ASSESSMENTS_URL` is configured. The outgoing payload is always rubric version "v2" and uses the skill-aligned format.

- Headers: may include `Authorization: Bearer <PERSIST_ASSESSMENTS_SECRET>` when configured, and `X-Request-Id` when available.
- Body shape:

```json
{
  "sessionId": "<string>",
  "groupId": "<string>",
  "rubricVersion": "v2",
  "summary": {
    "skillAssessments": [
      {
        "skillHash": "<sha256>",
        "level": 0.0,
        "metCriteria": ["..."],
        "unmetCriteria": ["..."],
        "feedback": ["..."]
      }
    ],
    "meta": {
      "provider": "<provider>",
      "model": "<model>",
      "skillsCount": 1
    }
  }
}
```

Notes:
- Only `skillHash` is persisted (not raw IDs). Hashing uses SHA-256 and can include an optional salt via `SKILL_HASH_SALT`.
- Legacy v1 persistence is removed. The `ASSESS_OUTPUT_V2` toggle has been deleted; persistence is always v2.

### GET /metrics
- Summary: Prometheus metrics endpoint for SQS + assessment worker instrumentation.
- Media type: text/plain; version=0.0.4

Metrics of interest:
- coachup_chat_ttft_seconds — time to first token for chat (Histogram)
- coachup_chat_total_seconds — total chat latency (Histogram)
- coachup_assessment_job_seconds — overall assessment job latency (Histogram)
- coachup_assessments_jobs_total — assessment job outcomes (Counter; labels: status="success|failure")
- coachup_assessment_skill_seconds — per-skill assessment latency (Histogram; labels: provider, model, rubric_version)
- coachup_assessment_skill_errors_total — per-skill assessment errors (Counter; labels: provider, model, reason)
- coachup_assessment_tokens_total — aggregated assessment tokens (Counter; labels: direction="in|out", provider, model, rubric_version)
- coachup_assessment_cost_usd_total — aggregated assessment spend in USD (Counter; labels: provider, model, rubric_version)
- coachup_assessments_enqueue_latency_seconds — latency from enqueue to dequeue (Histogram)
- coachup_assessments_retries_total — assessment retries (Counter)
- coachup_assessments_queue_depth — in-memory queue depth (Gauge; label: provider)

Notes:
- p95 values are typically computed via histogram_quantile over the *_seconds_bucket series in Grafana/Prometheus.
- Tokens/cost are emitted when provider metadata includes tokensIn/tokensOut/costUsd. For per-skill fan-out, values are aggregated across successful skills.

### GET /health
- Summary: healthcheck endpoint.

## Provider and model selection

The service selects providers and models via environment variables with sensible defaults and safe fallbacks.

- AI_PROVIDER — global default provider (mock | google | openrouter). Defaults to mock.
- Per-role overrides (take precedence over AI_PROVIDER):
  - AI_PROVIDER_CHAT, AI_PROVIDER_CLASSIFIER, AI_PROVIDER_ASSESS
- Feature toggles:
  - AI_CHAT_ENABLED, AI_CLASSIFIER_ENABLED (plus DISABLE_CLASSIFIER), AI_ASSESS_ENABLED
- Model hints (optional; provider may apply its own default):
  - AI_CHAT_MODEL, AI_CLASSIFIER_MODEL, AI_ASSESS_MODEL

Behavior and fallbacks:
- Factories in `app/providers/factory.py` honor explicit overrides and env flags.
- If a provider module or API key is unavailable, the factory returns a Mock client.
- Runtime call sites also catch provider errors and fall back to mock/deterministic behavior.

Example minimal configs:

```bash
# All mock (default)
AI_PROVIDER=mock

# Google chat only
AI_PROVIDER=mock
AI_PROVIDER_CHAT=google
AI_CHAT_ENABLED=1
AI_CHAT_MODEL=gemini-1.5-flash
GOOGLE_API_KEY=...

# OpenRouter assessments only
AI_PROVIDER=openrouter
AI_ASSESS_ENABLED=1
AI_ASSESS_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=...
```

## Timeouts

Assessment calls enforce time limits:
- ASSESS_PER_SKILL_TIMEOUT_MS (default 8000)
- ASSESS_GROUP_TIMEOUT_MS (default 15000)

Per-skill errors (including timeouts) are captured in the summary and reflected in Prometheus metrics with reason labels.

## OpenAPI Spec
- FastAPI serves /openapi.json automatically.
- Snapshot: curl http://localhost:8000/openapi.json > docs/api/ai/openapi.json

## Changelog
- 2025-08-26: Documented provider/model selection env flags and per-skill/group assessment timeouts; added assessment tokens/cost metrics and observability notes; updated examples.
- 2025-08-21: Updated endpoints to assessments run/get; documented expanded summary payload (categories, scores, meta, rubricVersion, rubricKeyPoints).
- 2025-08-25: Documented per-skill fan-out, tracked skill headers, and `summary.skillAssessments` schema.
- 2025-08-25: Persistence is now v2-only; added outgoing payload schema and removed references to `ASSESS_OUTPUT_V2`.
- 2025-08-19: Initial stub added.
