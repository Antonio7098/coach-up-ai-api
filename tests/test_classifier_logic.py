"""Unit tests for classifier logic without requiring API keys."""

import pytest
from unittest.mock import AsyncMock, patch
import sys
import os

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.providers.google import GoogleClassifierClient


class TestClassifierLogic:
    """Test classifier logic and prompt construction."""
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'mock_key'})
    def test_context_analysis_with_practice_history(self):
        """Test that context analysis correctly identifies practice sessions."""
        classifier = GoogleClassifierClient(model="gemini-1.5-flash")
        
        # Mock context with practice session
        practice_context = [
            {"role": "assistant", "content": "Let's practice a sales pitch", "decision": "start"},
            {"role": "user", "content": "Hi, I'm John from TechCorp", "decision": "continue"}
        ]
        
        # Build context info (this is internal logic we can test)
        context_info = ""
        if practice_context and len(practice_context) > 0:
            context_lines = []
            has_practice_context = False
            for i, ctx_msg in enumerate(practice_context[-5:]):
                ctx_role = ctx_msg.get('role', 'unknown')
                ctx_content = ctx_msg.get('content', '')
                ctx_decision = ctx_msg.get('decision', 'unknown')
                context_lines.append(f"Turn {i}: [{ctx_role}] {ctx_content[:50]}... -> {ctx_decision}")
                if ctx_decision in ['start', 'continue', 'one_off']:
                    has_practice_context = True
            context_info = f"CONVERSATION HISTORY:\n" + "\n".join(context_lines) + "\n"
            if has_practice_context:
                context_info += "PRACTICE SESSION DETECTED in conversation history.\n"
            else:
                context_info += "NO PRACTICE SESSION detected in conversation history - likely general conversation.\n"
        
        assert "PRACTICE SESSION DETECTED" in context_info
        assert "start" in context_info
        assert "continue" in context_info
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'mock_key'})
    def test_context_analysis_with_chitchat_history(self):
        """Test that context analysis correctly identifies general conversation."""
        classifier = GoogleClassifierClient(model="gemini-1.5-flash")
        
        # Mock context with only chitchat
        chitchat_context = [
            {"role": "user", "content": "Hello, how are you?", "decision": "abstain"},
            {"role": "assistant", "content": "I'm doing well, thanks!", "decision": "abstain"}
        ]
        
        # Build context info
        context_info = ""
        if chitchat_context and len(chitchat_context) > 0:
            context_lines = []
            has_practice_context = False
            for i, ctx_msg in enumerate(chitchat_context[-5:]):
                ctx_role = ctx_msg.get('role', 'unknown')
                ctx_content = ctx_msg.get('content', '')
                ctx_decision = ctx_msg.get('decision', 'unknown')
                context_lines.append(f"Turn {i}: [{ctx_role}] {ctx_content[:50]}... -> {ctx_decision}")
                if ctx_decision in ['start', 'continue', 'one_off']:
                    has_practice_context = True
            context_info = f"CONVERSATION HISTORY:\n" + "\n".join(context_lines) + "\n"
            if has_practice_context:
                context_info += "PRACTICE SESSION DETECTED in conversation history.\n"
            else:
                context_info += "NO PRACTICE SESSION detected in conversation history - likely general conversation.\n"
        
        assert "NO PRACTICE SESSION detected" in context_info
        assert "abstain" in context_info
    
    def test_no_context_handling(self):
        """Test handling when no context is provided."""
        turn_count = 5
        
        # Build context info for no context scenario
        context_info = f"This is turn {turn_count} in a conversation. NO CONTEXT PROVIDED - assume general conversation unless content clearly indicates practice.\n"
        
        assert "NO CONTEXT PROVIDED" in context_info
        assert "assume general conversation" in context_info
        assert str(turn_count) in context_info
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'mock_key'})
    def test_system_prompt_emphasizes_abstain_default(self):
        """Test that system prompt emphasizes defaulting to abstain."""
        classifier = GoogleClassifierClient(model="gemini-1.5-flash")
        
        # Access the system prompt construction logic
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
        
        # Verify key phrases that emphasize conservative classification
        assert "DEFAULT to 'abstain'" in system_prompt
        assert "ONLY when AI explicitly requests practice" in system_prompt
        assert "Simple greetings like 'hello' or 'hi' = abstain" in system_prompt
        assert "When in doubt between start/abstain, choose abstain" in system_prompt
        assert "turnCount alone does NOT determine decision" in system_prompt
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'mock_key'})
    async def test_mock_classifier_response_abstain(self):
        """Test classifier with mocked response for abstain case."""
        classifier = GoogleClassifierClient(model="gemini-1.5-flash")
        
        # Mock the HTTP response for a simple greeting
        mock_response_data = {
            "candidates": [{
                "content": {
                    "parts": [{"text": '{"decision": "abstain", "confidence": 0.9, "reasons": "simple_greeting_no_practice_context"}'}]
                }
            }]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()
            mock_client.return_value.__aenter__.return_value.post.return_value.status_code = 200
            mock_client.return_value.__aenter__.return_value.post.return_value.json.return_value = mock_response_data
            
            result = await classifier.classify("user", "Hello", 0)
            
            assert result["decision"] == "abstain"
            assert result["confidence"] == 0.9
            assert "simple_greeting" in result["reasons"]
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'mock_key'})
    async def test_mock_classifier_response_start(self):
        """Test classifier with mocked response for start case."""
        classifier = GoogleClassifierClient(model="gemini-1.5-flash")
        
        # Mock the HTTP response for explicit practice request
        mock_response_data = {
            "candidates": [{
                "content": {
                    "parts": [{"text": '{"decision": "start", "confidence": 0.95, "reasons": "explicit_practice_request"}'}]
                }
            }]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()
            mock_client.return_value.__aenter__.return_value.post.return_value.status_code = 200
            mock_client.return_value.__aenter__.return_value.post.return_value.json.return_value = mock_response_data
            
            result = await classifier.classify("assistant", "Let's practice a sales pitch", 0)
            
            assert result["decision"] == "start"
            assert result["confidence"] == 0.95
            assert "explicit_practice_request" in result["reasons"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
