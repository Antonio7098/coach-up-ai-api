import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from app.providers.google import GoogleChatClient


class TestGoogleNetworkResilience:
    """Test network error handling and resilience in Google provider"""

    @pytest.fixture
    def google_client(self):
        """Create a Google client for testing"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            client = GoogleChatClient(model="gemini-test")
            # Reset counters for clean test state
            client._network_errors = 0
            client._total_requests = 0
            return client

    @pytest.mark.asyncio
    async def test_dns_resolution_error_with_retry(self, google_client):
        """Test that DNS resolution errors trigger retry logic"""
        # Mock httpx to simulate DNS resolution failure
        with patch('httpx.AsyncClient.stream') as mock_stream:
            # First two calls fail with DNS error, third succeeds
            mock_stream.side_effect = [
                httpx.ConnectError("Temporary failure in name resolution"),
                httpx.ConnectError("Temporary failure in name resolution"),
                self._create_mock_response_stream()
            ]

            # Collect yielded tokens
            tokens = []
            async for token in google_client.stream_chat("test prompt", request_id="test_123"):
                tokens.append(token)

            # Verify we got tokens (fallback worked)
            assert len(tokens) > 0

            # Verify retry attempts were made (3 total: initial + 2 retries)
            assert mock_stream.call_count == 3

            # Verify network error was tracked (2 retries = 2 errors counted)
            assert google_client._network_errors == 2

    @pytest.mark.asyncio
    async def test_connection_timeout_with_retry(self, google_client):
        """Test that connection timeouts trigger retry logic"""
        with patch('httpx.AsyncClient.stream') as mock_stream:
            # First call times out, second succeeds
            mock_stream.side_effect = [
                httpx.ConnectTimeout("Connection timed out"),
                self._create_mock_response_stream()
            ]

            tokens = []
            async for token in google_client.stream_chat("test prompt", request_id="test_456"):
                tokens.append(token)

            assert len(tokens) > 0
            assert mock_stream.call_count == 2
            assert google_client._network_errors == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_fallback(self, google_client):
        """Test that when all retries are exhausted, fallback mechanism is used"""
        with patch('httpx.AsyncClient.stream') as mock_stream, \
             patch('httpx.AsyncClient.post') as mock_post:

            # All stream attempts fail
            mock_stream.side_effect = httpx.ConnectError("Network unreachable")

            # Mock successful fallback response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Fallback response"}]
                    }
                }]
            }
            mock_post.return_value = mock_response

            tokens = []
            async for token in google_client.stream_chat("test prompt", request_id="test_789"):
                tokens.append(token)

            # Should get fallback response
            assert "Fallback response" in "".join(tokens)
            # All 3 attempts failed (initial + 2 retries)
            assert google_client._network_errors == 3

    @pytest.mark.asyncio
    async def test_non_network_errors_no_retry(self, google_client):
        """Test that non-network errors don't trigger retries"""
        with patch('httpx.AsyncClient.stream') as mock_stream:
            # API key error (non-network) - should not retry
            mock_stream.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized", request=MagicMock(), response=MagicMock()
            )

            with pytest.raises(httpx.HTTPStatusError):
                async for _ in google_client.stream_chat("test prompt"):
                    pass

            # Should only try once (no retries for non-network errors)
            assert mock_stream.call_count == 1

    def test_network_health_logging(self, google_client):
        """Test that network health metrics are logged periodically"""
        # Simulate multiple requests with some network errors
        google_client._total_requests = 9  # One short of logging threshold
        google_client._network_errors = 2

        # This should trigger health logging
        with patch('app.providers.google.logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            google_client._log_network_health("test_req")

            # Verify health metrics were logged
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            health_data = json.loads(call_args)

            assert health_data["event"] == "google_network_health"
            assert health_data["total_requests"] == 10  # Incremented
            assert health_data["network_errors"] == 2
            assert health_data["error_rate_percent"] == 20.0

    def _create_mock_response_stream(self):
        """Helper to create a mock streaming response"""
        async def mock_aiter_lines():
            # Simulate SSE-like response
            yield 'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}\n'
            yield 'data: {"candidates":[{"content":{"parts":[{"text":" world"}]}}]}\n'
            yield 'data: [DONE]\n'

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_context = MagicMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)

        return mock_stream_context

    @pytest.mark.asyncio
    async def test_environment_variable_configuration(self, google_client):
        """Test that environment variables properly configure timeouts and retries"""
        with patch.dict(os.environ, {
            "GOOGLE_NETWORK_RETRY_COUNT": "5",
            "GOOGLE_RETRY_DELAY_SECONDS": "2.0",
            "AI_HTTP_CONNECT_TIMEOUT_SECONDS": "15",
            "AI_HTTP_READ_TIMEOUT_SECONDS": "45"
        }), patch('httpx.AsyncClient.stream') as mock_stream:

            mock_stream.side_effect = httpx.ConnectError("Network error")

            # This should use custom retry count
            with pytest.raises(httpx.ConnectError):
                async for _ in google_client.stream_chat("test prompt"):
                    pass

            # Should have tried 6 times (1 initial + 5 retries)
            assert mock_stream.call_count == 6
