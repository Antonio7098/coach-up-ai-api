import os
import json
import hashlib
from typing import AsyncIterator, Optional, List, Dict, Any
import httpx

from .base import ChatClient, ClassifierClient, AssessClient


class OpenRouterChatClient(ChatClient):
    provider_name: str = "openrouter"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_CHAT_MODEL") or "openai/gpt-4o-mini")
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter provider")
        # HTTP settings
        self._api_key = api_key
        # Default 30s; can override via AI_HTTP_TIMEOUT_SECONDS or OPENROUTER_TIMEOUT_SECONDS
        try:
            self._timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS") or os.getenv("AI_HTTP_TIMEOUT_SECONDS") or 30)
        except Exception:
            self._timeout = 30.0
        # Origin metadata (optional but recommended by OpenRouter)
        self._referer = os.getenv("PUBLIC_APP_ORIGIN", "http://localhost:3000").strip() or "http://localhost:3000"
        self._title = os.getenv("OPENROUTER_APP_TITLE", "CoachUp AI API").strip() or "CoachUp AI API"

    async def stream_chat(self, prompt: str, system: Optional[str] = None, request_id: Optional[str] = None) -> AsyncIterator[str]:
        # Implement streaming via OpenRouter's SSE API compatible with OpenAI format
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "coach-up-ai-api/0.1.0 (+https://github.com/)",
            # Optional metadata for OpenRouter analytics
            "HTTP-Referer": self._referer,
            "X-Title": self._title,
        }
        if request_id:
            headers["X-Request-Id"] = request_id
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        # Optional cap on response length for concise coaching replies
        try:
            max_tokens_env = os.getenv("AI_CHAT_MAX_TOKENS", "120").strip()
            max_tokens = int(max_tokens_env) if max_tokens_env else 120
            if max_tokens > 0:
                payload["max_tokens"] = max_tokens
        except Exception:
            pass

        # Use a short-lived AsyncClient per request to ensure proper cleanup
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    # Raise to let caller handle fallback
                    text = await resp.aread()
                    raise RuntimeError(f"OpenRouter error {resp.status_code}: {text!r}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    # SSE comment or keepalive
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        # If we can't parse JSON, skip the line
                        continue
                    # OpenAI-style delta streaming
                    try:
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        token = delta.get("content")
                        if token is None:
                            # Some providers use different field names
                            token = choices[0].get("text") or ""
                        if token:
                            yield token
                    except Exception:
                        # Skip malformed chunks
                        continue


class OpenRouterClassifierClient(ClassifierClient):
    provider_name: str = "openrouter"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_CLASSIFIER_MODEL") or "openai/gpt-4o-mini")
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter provider")
        self._api_key = api_key
        try:
            self._timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS") or os.getenv("AI_HTTP_TIMEOUT_SECONDS") or 30)
        except Exception:
            self._timeout = 30.0
        self._referer = os.getenv("PUBLIC_APP_ORIGIN", "http://localhost:3000").strip() or "http://localhost:3000"
        self._title = os.getenv("OPENROUTER_APP_TITLE", "CoachUp AI API").strip() or "CoachUp AI API"

    async def classify(self, role: str, content: str, turn_count: int, request_id: Optional[str] = None) -> dict:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "coach-up-ai-api/0.1.0 (+https://github.com/)",
            "HTTP-Referer": self._referer,
            "X-Title": self._title,
        }
        if request_id:
            headers["X-Request-Id"] = request_id
        system_prompt = (
            "You are a message boundary classifier for multi-turn interactions. "
            "Decide whether the current message marks the 'start' of an interaction, "
            "'continue' of an ongoing one, 'end' of an interaction, or a 'one_off' single message. "
            "Output strict JSON with keys: decision in ['start','continue','end','one_off','abstain'], "
            "confidence as a float 0..1, and reasons as a short snake_case string."
        )
        user_prompt = (
            f"role: {role}\nturnCount: {turn_count}\ncontent:\n" + content
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            # Prefer JSON if model supports it; benign for others
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code >= 400:
                    raise RuntimeError(f"OpenRouter classify error {resp.status_code}: {resp.text}")
                data = resp.json()
        except Exception as e:
            return {"decision": "abstain", "confidence": 0.0, "reasons": f"provider_error:{type(e).__name__}"}

        try:
            msg = (((data or {}).get("choices") or [{}])[0].get("message") or {})
            content_text = (msg.get("content") or "").strip()
            # Strip markdown JSON fences if present
            if content_text.startswith("```"):
                # naive fence removal
                content_text = content_text.strip("`")
                # After removing backticks, try to locate JSON braces
            # Try to parse JSON directly
            obj = json.loads(content_text)
            decision = str(obj.get("decision", "abstain")).lower()
            confidence = float(obj.get("confidence", 0.0))
            reasons = str(obj.get("reasons", "llm_output")).strip() or "llm_output"
            # Guard rails
            if decision not in ("start", "continue", "end", "one_off", "abstain"):
                decision = "abstain"
            if not (0.0 <= confidence <= 1.0):
                confidence = 0.0
            return {"decision": decision, "confidence": confidence, "reasons": reasons}
        except Exception:
            return {"decision": "abstain", "confidence": 0.0, "reasons": "parse_error"}


class OpenRouterAssessClient(AssessClient):
    provider_name: str = "openrouter"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_ASSESS_MODEL") or "openai/gpt-4o-mini")
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter provider")
        self._api_key = api_key
        # HTTP timeout (seconds)
        try:
            self._timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS") or os.getenv("AI_HTTP_TIMEOUT_SECONDS") or 30)
        except Exception:
            self._timeout = 30.0
        # Optional origin metadata
        self._referer = os.getenv("PUBLIC_APP_ORIGIN", "http://localhost:3000").strip() or "http://localhost:3000"
        self._title = os.getenv("OPENROUTER_APP_TITLE", "CoachUp AI API").strip() or "CoachUp AI API"

    async def assess(
        self,
        transcript: list[dict],
        rubric_version: str = "v2",
        request_id: Optional[str] = None,
        skill: Optional[Dict[str, Any]] = None,
        target_role: Optional[str] = "user",
    ) -> dict:
        """Call OpenRouter chat completions to produce an assessment summary.

        Expected return keys:
          - rubricVersion: str
          - categories: List[str]
          - scores: Dict[str, float]  (values in [0,1])
          - highlights: List[str]
          - recommendations: List[str]
          - meta: Dict[str, Any] (optional)
        """
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "coach-up-ai-api/0.1.0 (+https://github.com/)",
            # Optional metadata for OpenRouter analytics
            "HTTP-Referer": self._referer,
            "X-Title": self._title,
        }
        if request_id:
            headers["X-Request-Id"] = request_id
        # Propagate skill observability headers when available
        if skill:
            sid = str(skill.get("id") or "").strip()
            if sid:
                headers["X-Tracked-Skill-Id"] = sid
                try:
                    headers["X-Tracked-Skill-Id-Hash"] = hashlib.sha256(sid.encode("utf-8")).hexdigest()
                except Exception:
                    pass

        # Build a concise representation of the transcript
        def _format_event(e: Dict[str, Any]) -> str:
            role = (e.get("role") or "").strip()
            content = (e.get("content") or "").strip()
            return f"{role}: {content}"

        # Filter to only the target role if specified (default: user)
        _tgt = (target_role or "").strip()
        if _tgt:
            filtered_events = [ev for ev in (transcript or []) if (ev.get("role") or "").strip() == _tgt]
        else:
            filtered_events = transcript or []

        convo_lines: List[str] = [_format_event(ev) for ev in filtered_events]
        convo_text = "\n".join(convo_lines)

        # Include optional skill context in the instruction to focus evaluation
        skill_line = ""
        if skill:
            sname = str(skill.get("name") or "").strip()
            scategory = str(skill.get("category") or "").strip()
            sid = str(skill.get("id") or "").strip()
            parts = [p for p in [sname, scategory, sid] if p]
            if parts:
                skill_line = "\nFocus skill: " + " | ".join(parts)

        system_prompt = (
            "You are an expert coach conversation assessor. "
            "Given a short transcript, evaluate it using rubric '" + rubric_version + "'. "
            "Return STRICT JSON with keys: rubricVersion, categories, scores, highlights, recommendations, meta. "
            "- rubricVersion: echo the provided rubric version string. "
            "- categories: array of category names. "
            "- scores: object mapping each category to a float 0..1. "
            "- highlights: 1-3 short quotes or snippets. "
            "- recommendations: 2-3 short actionable tips. "
            "- meta: optional object with any additional details. "
            "Assess ONLY the messages from role '" + (_tgt or "user") + "'. Ignore other roles. "
            "Do not include any commentary outside of the JSON."
        ) + skill_line
        user_prompt = (
            "Transcript (role: content):\n" + convo_text
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            # Prefer structured JSON when supported
            "response_format": {"type": "json_object"},
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code >= 400:
                    raise RuntimeError(f"OpenRouter assess error {resp.status_code}: {resp.text}")
                data = resp.json()
        except Exception:
            # Fallback minimal structure to keep pipeline resilient
            return {
                "rubricVersion": rubric_version,
                "categories": [],
                "scores": {},
                "highlights": ["placeholder"],
                "recommendations": ["placeholder"],
                "meta": {"provider": "openrouter", "error": "request_failed"},
            }

        # Parse provider response
        try:
            msg = (((data or {}).get("choices") or [{}])[0].get("message") or {})
            content_text = (msg.get("content") or "").strip()
            # Remove markdown fences if present
            if content_text.startswith("```"):
                content_text = content_text.strip("`")
            obj: Dict[str, Any] = json.loads(content_text)

            out: Dict[str, Any] = {
                "rubricVersion": str(obj.get("rubricVersion") or rubric_version),
                "categories": list(obj.get("categories") or []),
                "scores": dict(obj.get("scores") or {}),
                "highlights": list(obj.get("highlights") or []),
                "recommendations": list(obj.get("recommendations") or []),
                "meta": dict(obj.get("meta") or {}),
            }
            # Guard-rail: coerce score values into [0,1]
            scores: Dict[str, Any] = out.get("scores", {})
            fixed_scores: Dict[str, float] = {}
            for k, v in scores.items():
                try:
                    x = float(v)
                except Exception:
                    continue
                if x < 0.0:
                    x = 0.0
                if x > 1.0:
                    x = 1.0
                fixed_scores[str(k)] = x
            out["scores"] = fixed_scores
            # Attach provider/meta and echo skill information when present
            out.setdefault("meta", {})
            try:
                out["meta"].setdefault("provider", self.provider_name)
                if request_id:
                    out["meta"]["requestId"] = request_id
                if target_role:
                    out["meta"]["targetRole"] = target_role
                if skill:
                    out["meta"]["skill"] = {k: skill.get(k) for k in ("id", "name", "category") if k in skill}
            except Exception:
                pass
            return out
        except Exception:
            return {
                "rubricVersion": rubric_version,
                "categories": [],
                "scores": {},
                "highlights": ["placeholder"],
                "recommendations": ["placeholder"],
                "meta": {"provider": "openrouter", "error": "parse_error"},
            }
