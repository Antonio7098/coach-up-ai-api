import os
import json
import logging
from typing import AsyncIterator, Optional, Any, Dict, List

import httpx

from .base import ChatClient, ClassifierClient, AssessClient


class GoogleChatClient(ChatClient):
    provider_name: str = "google"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_CHAT_MODEL") or "gemini-1.5-pro")
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Google provider")
        # TODO(SPR-005): initialize google generative ai client with api_key
        self._api_key = api_key

    async def stream_chat(self, prompt: str, system: Optional[str] = None, request_id: Optional[str] = None) -> AsyncIterator[str]:
        """Stream chat tokens using Google Generative Language API (Gemini).

        Endpoint: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key=API_KEY
        Docs: https://ai.google.dev/api/rest/v1beta/models/streamGenerateContent
        """
        api_key = self._api_key
        model = (self.model or "gemini-1.5-pro").strip()
        base_url = "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base_url}/models/{model}:streamGenerateContent?key={api_key}"

        # Build request payload per Google Generative Language API
        messages: List[Dict[str, Any]] = []
        if system:
            # v1beta supports systemInstruction
            # Per API, systemInstruction is a Content object; role is optional. Sending without role for compatibility.
            system_instruction = {"parts": [{"text": system}]}
        else:
            system_instruction = None
        user_content = {"role": "user", "parts": [{"text": prompt}]}

        payload: Dict[str, Any] = {
            "contents": [user_content],
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        # Optional cap on response length
        try:
            max_tokens_env = os.getenv("AI_CHAT_MAX_TOKENS", "120").strip()
            max_tokens = int(max_tokens_env) if max_tokens_env else 120
            if max_tokens > 0:
                payload["generationConfig"] = {"maxOutputTokens": max_tokens}
        except Exception:
            pass

        # Timeout configuration (seconds)
        try:
            timeout_s = float(os.getenv("AI_HTTP_TIMEOUT_SECONDS") or 30)
        except Exception:
            timeout_s = 30.0

        headers = {
            "Content-Type": "application/json",
            # Prefer SSE if served; Google returns NDJSON-like chunks. Accept both.
            "Accept": "text/event-stream, application/json",
        }
        if request_id:
            headers["X-Request-Id"] = request_id

        debug = os.getenv("GOOGLE_STREAM_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
        logger = logging.getLogger("coach_up.ai.google")

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code >= 400:
                        text = await resp.aread()
                        raise RuntimeError(f"Google stream error {resp.status_code}: {text!r}")
                    had_output = False
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if debug:
                            try:
                                logger.info("[google.stream] raw_line=%s", (line[:512] + ("..." if len(line) > 512 else "")))
                            except Exception:
                                pass
                        # Some servers send SSE-style 'data: ' prefixes — strip if present
                        if line.startswith("data: "):
                            line = line[len("data: ") : ]
                        # Ignore SSE control lines
                        if line.startswith(":") or line.startswith("event:" ):
                            continue
                        # Each line is expected to be a JSON object chunk
                        try:
                            obj = json.loads(line)
                        except Exception:
                            # Some servers may send non-JSON keepalives; skip
                            continue
                        # Chunk shape per API: { candidates: [ { content: { parts: [ { text } ] } } ] }
                        try:
                            candidates = obj.get("candidates") or []
                            if not candidates:
                                continue
                            content = (candidates[0].get("content") or {})
                            parts = content.get("parts") or []
                            for p in parts:
                                t = (p.get("text") or "")
                                if t:
                                    had_output = True
                                    yield t
                            # If finishReason present and not "STOP", we still just stop silently
                            # (caller controls overall stream lifecycle)
                        except Exception:
                            continue
                    # If stream finished with no tokens, try non-stream fallback once
                    if not had_output:
                        if debug:
                            try:
                                logger.info("[google.stream] no streamed tokens; attempting non-stream fallback")
                            except Exception:
                                pass
                        nonstream_url = f"{base_url}/models/{model}:generateContent?key={api_key}"
                        r = await client.post(nonstream_url, headers=headers, json=payload)
                        if r.status_code < 400:
                            data = r.json()
                            candidates = (data or {}).get("candidates") or []
                            if candidates:
                                content = (candidates[0].get("content") or {})
                                parts = content.get("parts") or []
                                for p in parts:
                                    t = (p.get("text") or "")
                                    if t:
                                        yield t
                                return
            except Exception as e:
                # Attempt non-stream fallback (single response) to salvage a reply
                try:
                    if debug:
                        try:
                            logger.exception("[google.stream] error during stream; attempting non-stream fallback: %s", e)
                        except Exception:
                            pass
                    nonstream_url = f"{base_url}/models/{model}:generateContent?key={api_key}"
                    r = await client.post(nonstream_url, headers=headers, json=payload)
                    if r.status_code >= 400:
                        raise RuntimeError(f"Google generateContent error {r.status_code}: {r.text}")
                    data = r.json()
                    candidates = (data or {}).get("candidates") or []
                    if candidates:
                        content = (candidates[0].get("content") or {})
                        parts = content.get("parts") or []
                        for p in parts:
                            t = (p.get("text") or "")
                            if t:
                                yield t
                        return
                except Exception:
                    pass
                # If all fails, bubble up to let caller handle fallback
                raise


class GoogleClassifierClient(ClassifierClient):
    provider_name: str = "google"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_CLASSIFIER_MODEL") or "gemini-1.5-pro")
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Google provider")
        self._api_key = api_key

    async def classify(self, role: str, content: str, turn_count: int, request_id: Optional[str] = None) -> dict:
        # TODO(SPR-005): implement real Google classification
        # Raise to let caller fall back to mock/heuristics during early scaffolding
        raise NotImplementedError("Google classifier not implemented yet")


class GoogleAssessClient(AssessClient):
    provider_name: str = "google"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_ASSESS_MODEL") or "gemini-1.5-pro")
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Google provider")
        self._api_key = api_key

    async def assess(self, transcript: list[dict], rubric_version: str = "v1", request_id: Optional[str] = None, skill: Optional[dict] = None) -> dict:
        # TODO(SPR-005): implement real Google assessment client
        raise NotImplementedError("Google assess not implemented yet")
