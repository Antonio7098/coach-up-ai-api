# Error Model and Codes

## Envelope
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

- code: machine-readable (e.g., "unauthorized", "rate_limit_exceeded", "validation_error").
- message: human-friendly, safe for logs.
- details: optional object with field errors or context.

## Common Codes
- unauthorized — missing/invalid auth
- forbidden — lacks permission
- not_found — resource missing
- validation_error — invalid input
- rate_limit_exceeded — too many requests
- provider_error — upstream model/provider failure
- timeout — operation timed out
- conflict — idempotency/version conflict

## Status Mapping
- 401 unauthorized, 403 forbidden, 404 not_found
- 409 conflict, 422 validation_error
- 429 rate_limit_exceeded
- 500 provider_error, 504 timeout
