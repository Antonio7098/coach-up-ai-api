SPECTRAL = npx -y @stoplight/spectral-cli
REDOCLY = npx -y @redocly/cli

.PHONY: help openapi-snapshot openapi-lint redoc-build

help:
	@echo "Targets:"
	@echo "  openapi-snapshot    Snapshot FastAPI OpenAPI to docs/api/ai/openapi.json"
	@echo "  openapi-lint        Lint OpenAPI with Spectral"
	@echo "  redoc-build         Build static HTML docs with Redocly"

openapi-snapshot:
	curl -sS http://localhost:8000/openapi.json > docs/api/ai/openapi.json
	@echo "Wrote docs/api/ai/openapi.json"

openapi-lint:
	$(SPECTRAL) lint --ruleset docs/api/.spectral.yaml docs/api/ai/openapi.json

redoc-build:
	$(REDOCLY) build-docs docs/api/ai/openapi.json -o docs/api/ai/reference.html
