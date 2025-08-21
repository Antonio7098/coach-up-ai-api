# Shared API Headers & Conventions

## Auth
- Service auth (Core → AI): verify Clerk JWT (issuer/audience, JWKS cache) or use a service secret for background jobs

## Request ID
- X-Request-Id: stable per client request; propagate caller → FastAPI → provider

## Content Types
- application/json for REST
- text/event-stream for SSE streaming endpoints

## Rate Limits
- AI API typically protected by upstream Core API limits; document if direct limits apply
- 429 with Retry-After header on limit exceeded

## Tracing
- Include requestId in all logs; optionally include hashed userId/sessionId/trackedSkillId/groupId
