import asyncio
import json
from types import SimpleNamespace

import pytest

# Import-or-skip for optional SQS deps
boto3 = pytest.importorskip(
    "boto3",
    reason="SQS tests require boto3. Install with: python -m pip install -r requirements.txt",
)
botocore = pytest.importorskip(
    "botocore",
    reason="SQS tests require botocore (installed transitively with boto3).",
)
from botocore.stub import Stubber

import app.main as main


def make_app_ns():
    return SimpleNamespace(
        state=SimpleNamespace(
            assessment_queue=asyncio.Queue(),
            assessment_results={},
            assessments_enqueued_ts={},
        )
    )


@pytest.mark.asyncio
async def test_enqueue_sqs_sends_message(monkeypatch):
    # Enable SQS
    main.USE_SQS = True
    main.AWS_REGION = "us-east-1"
    main.AWS_SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/coach-up-assessments.fifo"
    main.AWS_ENDPOINT_URL_SQS = ""

    client = boto3.client("sqs", region_name="us-east-1")
    stubber = Stubber(client)

    session_id = "s1"
    group_id = "g1"
    request_id = "r1"

    expected_params = {
        "QueueUrl": main.AWS_SQS_QUEUE_URL,
        "MessageBody": json.dumps({
            "sessionId": session_id,
            "groupId": group_id,
            "requestId": request_id,
        }),
        "MessageGroupId": session_id,
        "MessageDeduplicationId": f"{session_id}:{group_id}",
    }
    stubber.add_response(
        "send_message",
        {"MessageId": "mid-123", "MD5OfMessageBody": "abc"},
        expected_params,
    )

    stubber.activate()
    monkeypatch.setattr(main, "_get_sqs_client", lambda: client)

    app_ns = make_app_ns()

    await main._enqueue_assessment_job(app_ns, session_id, group_id, request_id)

    # No fallback enqueue occurred
    assert app_ns.state.assessment_queue.qsize() == 0

    stubber.assert_no_pending_responses()
    stubber.deactivate()


class FakeSQSClient:
    def __init__(self, messages):
        self._messages = list(messages)
        self.deleted = []
        self.visibility_changed = []

    # Methods used by worker
    def receive_message(self, QueueUrl, MaxNumberOfMessages, WaitTimeSeconds, VisibilityTimeout):
        if self._messages:
            return {"Messages": [self._messages.pop(0)]}
        return {"Messages": []}

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted.append(ReceiptHandle)
        return {}

    def change_message_visibility(self, QueueUrl, ReceiptHandle, VisibilityTimeout):
        self.visibility_changed.append((ReceiptHandle, VisibilityTimeout))
        return {}


@pytest.mark.asyncio
async def test_worker_sqs_success_deletes_message_and_updates_results(monkeypatch):
    # Enable SQS
    main.USE_SQS = True
    main.AWS_SQS_QUEUE_URL = "https://example.com/q.fifo"

    session_id = "sess-123"
    group_id = "grp-123"
    request_id = "req-123"

    msg = {
        "ReceiptHandle": "rh-1",
        "Body": json.dumps({
            "sessionId": session_id,
            "groupId": group_id,
            "requestId": request_id,
        }),
    }
    fake = FakeSQSClient([msg])

    monkeypatch.setattr(main, "_get_sqs_client", lambda: fake)

    app_ns = make_app_ns()

    # Run worker in background and cancel after one successful loop
    task = asyncio.create_task(main._assessments_worker_sqs(app_ns, 0))

    # Wait until results are written or timeout
    for _ in range(50):
        await asyncio.sleep(0.05)
        if session_id in app_ns.state.assessment_results:
            break
    # Also wait for delete to be called to avoid race with cancellation
    for _ in range(50):
        await asyncio.sleep(0.05)
        if fake.deleted:
            break
    # Cancel and await task; worker swallows CancelledError and exits gracefully
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=1)
    except asyncio.CancelledError:
        # Some platforms may still raise
        pass

    # Assert delete called and results present
    assert fake.deleted == ["rh-1"]
    assert session_id in app_ns.state.assessment_results
    res = app_ns.state.assessment_results[session_id]
    assert res.get("latestGroupId") == group_id
    assert "summary" in res


@pytest.mark.asyncio
async def test_worker_sqs_error_changes_visibility(monkeypatch):
    # Enable SQS and force job error
    main.USE_SQS = True
    main.AWS_SQS_QUEUE_URL = "https://example.com/q.fifo"

    session_id = "sess-err"
    group_id = "grp-err"

    msg = {
        "ReceiptHandle": "rh-err",
        "Body": json.dumps({
            "sessionId": session_id,
            "groupId": group_id,
            "requestId": None,
        }),
    }
    fake = FakeSQSClient([msg])

    async def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "_get_sqs_client", lambda: fake)
    monkeypatch.setattr(main, "_run_assessment_job", _raise)

    app_ns = make_app_ns()

    task = asyncio.create_task(main._assessments_worker_sqs(app_ns, 0))

    # Wait until visibility change called or timeout
    for _ in range(50):
        await asyncio.sleep(0.05)
        if fake.visibility_changed:
            break
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=1)
    except asyncio.CancelledError:
        # Some platforms may still raise
        pass

    assert fake.visibility_changed, "Expected change_message_visibility to be called on error"
