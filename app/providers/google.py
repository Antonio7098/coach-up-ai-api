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
                        raw = await resp.aread()
                        body = raw[:1024]
                        try:
                            logger.error(
                                json.dumps({
                                    "event": "google_stream_http_error",
                                    "status": resp.status_code,
                                    "headers": dict(resp.headers),
                                    "body": body.decode(errors="replace"),
                                    "model": model,
                                })
                            )
                        except Exception:
                            pass
                        raise RuntimeError(f"Google stream error {resp.status_code}: {body!r}")
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
                        else:
                            try:
                                logger.error(json.dumps({
                                    "event": "google_nonstream_http_error",
                                    "status": r.status_code,
                                    "headers": dict(r.headers),
                                    "body": (r.text or "")[:1024],
                                    "model": model,
                                }))
                            except Exception:
                                pass
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
                        try:
                            logger.error(json.dumps({
                                "event": "google_fallback_http_error",
                                "status": r.status_code,
                                "headers": dict(r.headers),
                                "body": (r.text or "")[:1024],
                                "model": model,
                            }))
                        except Exception:
                            pass
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

    async def classify(self, role: str, content: str, turn_count: int, request_id: Optional[str] = None, context: Optional[list] = None) -> dict:
        """LLM-based classifier using Google Gemini to detect speech practice and skill improvement content."""
        api_key = self._api_key
        model = (self.model or "gemini-1.5-flash").strip()
        base_url = "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base_url}/models/{model}:generateContent?key={api_key}"
        
        # System prompt focused on detecting speech practice and skill improvement (not general chitchat)
        system_prompt = (
            "You are a message classifier for a speech coaching app that detects when users are practicing communication skills. "
            "IDENTIFY AS SPEECH PRACTICE: "
            "- AI explicitly requesting practice ('greet me', 'give me a pitch', 'practice your presentation', 'let's roleplay') "
            "- Roleplay scenarios with clear practice intent (sales calls, presentations, customer service) "
            "- Actual speech attempts with filler words (um, uh, er, like) indicating practice delivery "
            "- Formal presentations, pitch practice, public speaking rehearsal "
            "- Users explicitly responding to coaching prompts or practice requests from AI "
            "- Content clearly delivered as if speaking to an audience or practicing delivery "
            "IGNORE AS CHITCHAT (must abstain): "
            "- Casual questions about weather, personal life, general topics "
            "- Simple greetings without explicit practice context ('hello', 'hi there', 'good morning') "
            "- Administrative requests, casual thank-yous, or general conversation "
            "- Questions about the app, technical issues, or general help "
            "- Social pleasantries that don't involve skill practice "
            "DECISIONS: "
            "'start' = ONLY when AI explicitly requests practice OR user explicitly begins practice with clear intent "
            "'continue' = ongoing practice conversation where previous context shows active practice session "
            "'end' = AI giving final feedback ('well done', 'great job', 'let's practice X next') OR clear session conclusion "
            "'one_off' = single standalone practice attempt with no multi-turn context expected "
            "'abstain' = general conversation, casual greetings, polite thanks, non-practice content "
            "CRITICAL RULES: "
            "- DEFAULT to 'abstain' unless there's CLEAR practice intent "
            "- Simple greetings like 'hello' or 'hi' = abstain (not start) "
            "- Only classify as 'start' if there's explicit practice language or coaching request "
            "- Only classify as 'continue' if previous context shows active practice session "
            "- turnCount alone does NOT determine decision - content and context matter most "
            "- When in doubt between start/abstain, choose abstain "
            "Return JSON: {\"decision\": \"start|continue|end|one_off|abstain\", \"confidence\": 0.0-1.0, \"reasons\": \"explanation\"}"
        )
        
        # Build conversation context from previous messages
        context_info = ""
        if context and len(context) > 0:
            context_lines = []
            has_practice_context = False
            for i, ctx_msg in enumerate(context[-5:]):  # Last 5 messages
                ctx_role = ctx_msg.get('role', 'unknown')
                ctx_content = ctx_msg.get('content', '')
                ctx_decision = ctx_msg.get('decision', 'unknown')
                context_lines.append(f"Turn {i}: [{ctx_role}] {ctx_content[:50]}... -> {ctx_decision}")
                # Check if there's actual practice context
                if ctx_decision in ['start', 'continue', 'one_off']:
                    has_practice_context = True
            context_info = f"CONVERSATION HISTORY:\n" + "\n".join(context_lines) + "\n"
            if has_practice_context:
                context_info += "PRACTICE SESSION DETECTED in conversation history.\n"
            else:
                context_info += "NO PRACTICE SESSION detected in conversation history - likely general conversation.\n"
        elif turn_count > 0:
            context_info = f"This is turn {turn_count} in a conversation. NO CONTEXT PROVIDED - assume general conversation unless content clearly indicates practice.\n"
        
        user_prompt = f"{context_info}CURRENT MESSAGE:\nrole: {role}\nturnCount: {turn_count}\ncontent:\n{content}"
        
        # Build request payload for Google Generative Language API
        # Use systemInstruction and enforce JSON response to increase parsing reliability across models (incl. Gemini 2.5 Flash)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "candidateCount": 1,
                "maxOutputTokens": 200,
                "responseMimeType": "application/json",
            },
            # v1beta supports systemInstruction as a Content object
            "systemInstruction": {"parts": [{"text": system_prompt}]},
        }
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "coach-up-ai-api/0.1.0",
            }
            if request_id:
                headers["X-Request-Id"] = request_id
                
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code >= 400:
                    try:
                        logging.getLogger("coach_up.ai.google").error(json.dumps({
                            "event": "google_classify_http_error",
                            "status": resp.status_code,
                            "headers": dict(resp.headers),
                            "body": (resp.text or "")[:1024],
                            "model": self.model,
                        }))
                    except Exception:
                        pass
                    raise RuntimeError(f"Google classify error {resp.status_code}: {resp.text}")
                data = resp.json()
        except Exception as e:
            return {"decision": "abstain", "confidence": 0.0, "reasons": f"provider_error:{type(e).__name__}"}
        
        try:
            # Parse Google API response structure
            candidates = data.get("candidates", [])
            if not candidates:
                return {"decision": "abstain", "confidence": 0.0, "reasons": "no_candidates"}
            
            content_obj = candidates[0].get("content", {})
            parts = content_obj.get("parts", [])
            if not parts:
                return {"decision": "abstain", "confidence": 0.0, "reasons": "no_parts"}
            
            content_text = parts[0].get("text", "").strip()
            # Optional debug logging of raw content for observability (truncated)
            try:
                if os.getenv("GOOGLE_CLASSIFY_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on"):
                    logging.getLogger("coach_up.ai.google").info(json.dumps({
                        "event": "google_classify_raw",
                        "model": model,
                        "preview": content_text[:800],
                    }))
            except Exception:
                pass
            
            # Clean up common markdown artifacts
            if content_text.startswith("```"):
                # Remove typical fenced code blocks like ```json ... ```
                lines = content_text.split('\n')
                if len(lines) > 2:
                    # Drop first and last lines when they are fences
                    if lines[0].lstrip().startswith("```") and lines[-1].lstrip().startswith("```"):
                        content_text = '\n'.join(lines[1:-1])
                content_text = content_text.strip()

            # If extra prose is around JSON, try to extract the first top-level JSON object
            if not (content_text.startswith("{") and content_text.rstrip().endswith("}")):
                try:
                    start = content_text.find("{")
                    end = content_text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        content_text = content_text[start:end+1]
                except Exception:
                    pass
            
            # Try to parse as JSON
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


class GoogleAssessClient(AssessClient):
    provider_name: str = "google"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model or os.getenv("AI_ASSESS_MODEL") or "gemini-1.5-pro")
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Google provider")
        self._api_key = api_key
        # HTTP timeout (seconds)
        try:
            self._timeout = float(os.getenv("GOOGLE_TIMEOUT_SECONDS") or os.getenv("AI_HTTP_TIMEOUT_SECONDS") or 30)
        except Exception:
            self._timeout = 30.0

    async def assess(
        self,
        transcript: list[dict],
        rubric_version: str = "v2",
        request_id: Optional[str] = None,
        skill: Optional[Dict[str, Any]] = None,
        target_role: Optional[str] = "user",
    ) -> dict:
        """Call Google Gemini API to produce an assessment summary.

        Expected return keys:
          - rubricVersion: str
          - categories: List[str]
          - scores: Dict[str, float]  (values in [0,1])
          - highlights: List[str]
          - recommendations: List[str]
          - meta: Dict[str, Any] (optional)
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "coach-up-ai-api/0.1.0 (+https://github.com/)",
        }
        if request_id:
            headers["X-Request-Id"] = request_id
        
        # Propagate skill observability headers when available
        if skill:
            sid = str(skill.get("id") or "").strip()
            if sid:
                headers["X-Tracked-Skill-Id"] = sid
                try:
                    import hashlib
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

        # Build detailed skill context including level, criteria, and rubric
        skill_context = ""
        if skill:
            sname = str(skill.get("name") or "").strip()
            scategory = str(skill.get("category") or "").strip()
            sid = str(skill.get("id") or "").strip()
            current_level = skill.get("currentLevel", 0)
            
            # Include skill metadata
            skill_parts = [p for p in [sname, scategory, sid] if p]
            if skill_parts:
                skill_context += f"\n\nSKILL FOCUS: {' | '.join(skill_parts)}"
                skill_context += f"\nCURRENT LEVEL: {current_level}/10"
            
            # Include level criteria and examples if available
            criteria = skill.get("criteria", {})
            if criteria:
                skill_context += "\n\nLEVEL CRITERIA:"
                for level, desc in criteria.items():
                    skill_context += f"\n  Level {level}: {desc}"
            
            # Include rubric details if available
            rubric = skill.get("rubric", {})
            if rubric:
                skill_context += "\n\nRUBRIC DETAILS:"
                for category, details in rubric.items():
                    skill_context += f"\n  {category}: {details}"
            
            # Include examples if available
            examples = skill.get("examples", {})
            if examples:
                skill_context += "\n\nEXAMPLES:"
                for level, example in examples.items():
                    skill_context += f"\n  Level {level}: {example}"

        system_prompt = (
            "You are an expert speech coach assessor. "
            f"Given a transcript, evaluate it using rubric '{rubric_version}' with focus on the specified skill. "
            "Determine the user's current skill level (1-10) based on the provided criteria and examples. "
            "Return STRICT JSON with keys: rubricVersion, level, highlights, recommendations, rubricKeyPoints, meta. "
            "- rubricVersion: echo the provided rubric version string. "
            "- level: integer from 1-10 representing the skill level achieved based on criteria. "
            "- highlights: 2-4 specific quotes or behaviors that stood out (both positive and areas for improvement). "
            "- recommendations: 3-5 specific, actionable tips for improvement to reach the next level. "
            "- rubricKeyPoints: 2-3 key criteria points that were met or not met in this assessment. "
            "- meta: object with skill progression insights and reasoning for the level assigned. "
            f"Assess ONLY the messages from role '{_tgt or 'user'}'. Ignore other roles. "
            "Compare the user's performance against the level criteria. Be precise in level determination - "
            "if they don't fully meet a level's criteria, assign the lower level. Justify your level choice in meta."
            + skill_context
        )
        
        user_prompt = f"Transcript to assess:\n{convo_text}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json"
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    params={"key": self._api_key}
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"Google assess error {resp.status_code}: {resp.text}")
                data = resp.json()
        except Exception:
            # Fallback minimal structure to keep pipeline resilient
            return {
                "rubricVersion": rubric_version,
                "level": 1,
                "highlights": ["Assessment temporarily unavailable"],
                "recommendations": ["Please try again later"],
                "rubricKeyPoints": ["Unable to assess due to technical error"],
                "meta": {"provider": "google", "modelId": self.model, "error": "request_failed"},
            }

        # Parse Google API response
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in response")
            
            content_obj = candidates[0].get("content", {})
            parts = content_obj.get("parts", [])
            if not parts:
                raise ValueError("No parts in response")
            
            content_text = parts[0].get("text", "").strip()
            
            # Clean up common markdown artifacts
            if content_text.startswith("```"):
                lines = content_text.split('\n')
                if len(lines) > 2:
                    content_text = '\n'.join(lines[1:-1])
            
            obj: Dict[str, Any] = json.loads(content_text)

            out: Dict[str, Any] = {
                "rubricVersion": str(obj.get("rubricVersion") or rubric_version),
                "level": int(obj.get("level") or 1),
                "highlights": list(obj.get("highlights") or []),
                "recommendations": list(obj.get("recommendations") or []),
                "rubricKeyPoints": list(obj.get("rubricKeyPoints") or []),
                "meta": dict(obj.get("meta") or {}),
            }
            
            # Guard-rail: coerce level into [1,10]
            level = out.get("level", 1)
            try:
                level = int(level)
                if level < 1:
                    level = 1
                if level > 10:
                    level = 10
            except Exception:
                level = 1
            out["level"] = level
            
            # Attach provider/meta and echo skill information when present
            out.setdefault("meta", {})
            try:
                out["meta"].setdefault("provider", self.provider_name)
                out["meta"].setdefault("modelId", self.model)
                if request_id:
                    out["meta"]["requestId"] = request_id
                if target_role:
                    out["meta"]["targetRole"] = target_role
                if skill:
                    out["meta"]["skill"] = {k: skill.get(k) for k in ("id", "name", "category", "currentLevel") if k in skill}
            except Exception:
                pass
            return out
            
        except Exception:
            return {
                "rubricVersion": rubric_version,
                "level": 1,
                "highlights": ["Assessment parsing failed"],
                "recommendations": ["Please try again"],
                "rubricKeyPoints": ["Unable to parse assessment response"],
                "meta": {"provider": "google", "modelId": self.model, "error": "parse_error"},
            }
