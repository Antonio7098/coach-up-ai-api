SPECTRAL = npx -y @stoplight/spectral-cli
REDOCLY = npx -y @redocly/cli

# LocalStack / AWS CLI config (used for SQS helper targets)
# Defaults assume LocalStack on localhost:4566; override via env if needed
AWS_ENDPOINT_URL_SQS ?= http://localhost:4566
SQS_QUEUE_NAME ?= coach-up-assessments.fifo
# Prefer awslocal if available, otherwise fall back to aws with endpoint-url
AWS_CLI := $(shell command -v awslocal >/dev/null 2>&1 && echo awslocal || echo aws --endpoint-url $(AWS_ENDPOINT_URL_SQS))

.PHONY: help openapi-snapshot openapi-lint redoc-build sqs-create-local sqs-list-local sqs-purge-local sqs-delete-local sqs-create-dlq-local sqs-attach-dlq-local

help:
	@echo "Targets:"
	@echo "  openapi-snapshot    Snapshot FastAPI OpenAPI to docs/api/ai/openapi.json"
	@echo "  openapi-lint        Lint OpenAPI with Spectral"
	@echo "  redoc-build         Build static HTML docs with Redocly"
	@echo "  sqs-create-local    Create FIFO SQS queue on LocalStack ($(SQS_QUEUE_NAME)); prints QueueUrl"
	@echo "  sqs-list-local      List SQS queues on LocalStack"
	@echo "  sqs-purge-local     Purge messages from the LocalStack FIFO queue ($(SQS_QUEUE_NAME))"
	@echo "  sqs-delete-local    Delete the LocalStack FIFO queue ($(SQS_QUEUE_NAME))"
	@echo "  sqs-create-dlq-local Create a FIFO DLQ for $(SQS_QUEUE_NAME) (name: $(DLQ_QUEUE_NAME))"
	@echo "  sqs-attach-dlq-local Attach DLQ redrive policy to $(SQS_QUEUE_NAME) with maxReceiveCount=5"

openapi-snapshot:
	curl -sS http://localhost:8000/openapi.json > docs/api/ai/openapi.json
	@echo "Wrote docs/api/ai/openapi.json"

openapi-lint:
	$(SPECTRAL) lint --ruleset docs/api/.spectral.yaml docs/api/ai/openapi.json

redoc-build:
	$(REDOCLY) build-docs docs/api/ai/openapi.json -o docs/api/ai/reference.html

# --- Local SQS (LocalStack) helpers ---
sqs-create-local:
	@echo "Creating FIFO queue $(SQS_QUEUE_NAME) on $(AWS_ENDPOINT_URL_SQS)..."
	$(AWS_CLI) sqs create-queue \
		--queue-name $(SQS_QUEUE_NAME) \
		--attributes '{"FifoQueue":"true","ContentBasedDeduplication":"false","ReceiveMessageWaitTimeSeconds":"20","VisibilityTimeout":"60"}'
	@echo "Queue URL:"
	$(AWS_CLI) sqs get-queue-url --queue-name $(SQS_QUEUE_NAME)
	@echo "Set AWS_SQS_QUEUE_URL in your .env to the QueueUrl above."

sqs-list-local:
	$(AWS_CLI) sqs list-queues || true

sqs-purge-local:
	@echo "Purging queue $(SQS_QUEUE_NAME) ..."
	$(AWS_CLI) sqs purge-queue --queue-url $$($(AWS_CLI) sqs get-queue-url --queue-name $(SQS_QUEUE_NAME) --query 'QueueUrl' --output text)

sqs-delete-local:
	@echo "Deleting queue $(SQS_QUEUE_NAME) ..."
	$(AWS_CLI) sqs delete-queue --queue-url $$($(AWS_CLI) sqs get-queue-url --queue-name $(SQS_QUEUE_NAME) --query 'QueueUrl' --output text)

# DLQ helpers (LocalStack)
# Derive a DLQ name by replacing trailing .fifo with -dlq.fifo (SQS only allows .fifo suffix; no other dots)
DLQ_QUEUE_NAME := $(patsubst %.fifo,%-dlq.fifo,$(SQS_QUEUE_NAME))

sqs-create-dlq-local:
	@echo "Creating FIFO DLQ $(DLQ_QUEUE_NAME) on $(AWS_ENDPOINT_URL_SQS)..."
	$(AWS_CLI) sqs create-queue \
		--queue-name $(DLQ_QUEUE_NAME) \
		--attributes '{"FifoQueue":"true","ContentBasedDeduplication":"false","ReceiveMessageWaitTimeSeconds":"20","VisibilityTimeout":"60"}'
	@echo "DLQ URL:"
	$(AWS_CLI) sqs get-queue-url --queue-name $(DLQ_QUEUE_NAME)

sqs-attach-dlq-local:
	@echo "Attaching DLQ $(DLQ_QUEUE_NAME) to $(SQS_QUEUE_NAME) with maxReceiveCount=5 ..."
	@MAIN_URL=$$($(AWS_CLI) sqs get-queue-url --queue-name $(SQS_QUEUE_NAME) --query 'QueueUrl' --output text); \
	DLQ_URL=$$($(AWS_CLI) sqs get-queue-url --queue-name $(DLQ_QUEUE_NAME) --query 'QueueUrl' --output text); \
	DLQ_ARN=$$($(AWS_CLI) sqs get-queue-attributes --queue-url $$DLQ_URL --attribute-names QueueArn --query 'Attributes.QueueArn' --output text); \
	ATTR=$$(printf 'RedrivePolicy={"deadLetterTargetArn":"%s","maxReceiveCount":"5"}' "$$DLQ_ARN"); \
	$(AWS_CLI) sqs set-queue-attributes --queue-url $$MAIN_URL --attributes "$$ATTR"; \
	echo "Attached DLQ $$DLQ_ARN to $$MAIN_URL"
