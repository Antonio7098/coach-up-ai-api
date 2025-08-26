# Shared API Headers & Conventions

## Auth
- Service auth (Core → AI): verify Clerk JWT (issuer/audience, JWKS cache) or use a service secret for background jobs

## Request ID
- X-Request-Id: stable per client request; propagate caller → FastAPI → provider

## Tracked Skill Headers (Provider Calls)
- X-Tracked-Skill-Id: the raw skill identifier (when available) propagated from the assessment job to the AI provider.
- X-Tracked-Skill-Id-Hash: a privacy-preserving hash (sha256 hex) of the skill identifier used for observability and tracing.
- Notes:
  - These headers are set by the AI API when calling external AI providers during per-skill assessments.
  - Client → AI API calls do not need to set these headers; they are derived from tracked skills configured server-side.

## Content Types
- application/json for REST
- text/event-stream for SSE streaming endpoints

## Rate Limits
- AI API typically protected by upstream Core API limits; document if direct limits apply
- 429 with Retry-After header on limit exceeded

## Tracing
- Include requestId in all logs; optionally include hashed userId/sessionId/trackedSkillId/groupId
- Per-skill assessment logs and metrics will include trackedSkillIdHash for correlation.
