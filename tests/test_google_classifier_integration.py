"""Integration tests for Google LLM classifier using real Gemini API calls.

These tests verify the classifier can:
1. Detect one-off speech practice interactions
2. Identify multi-turn conversation boundaries (start/continue/end)
3. Filter out general chitchat (abstain)
4. Handle context from previous interactions
"""

import asyncio
import pytest
import os
import sys
from typing import List, Dict, Any

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.providers.google import GoogleClassifierClient


class TestGoogleClassifierIntegration:
    """Real API integration tests for Google classifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create Google classifier client."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        return GoogleClassifierClient(model="gemini-1.5-flash")
    
    @pytest.mark.asyncio
    async def test_one_off_greeting_practice(self, classifier):
        """Test one-off greeting practice detection."""
        print("\n🧪 Testing one-off greeting practice...")
        
        # AI asks for greeting practice
        ai_request = "greet me in a warm and welcoming fashion"
        result1 = await classifier.classify("assistant", ai_request, 0)
        
        print(f"AI request: {result1}")
        assert result1["decision"] in ["start", "continue"], "AI should initiate practice"
        
        # User responds with greeting practice - provide context
        context = [{"role": "assistant", "content": ai_request, "decision": result1["decision"]}]
        user_response = "hello, how are you doing today? Welcome to our company!"
        result2 = await classifier.classify("user", user_response, 1, context=context)
        
        print(f"User greeting: {result2}")
        # User responding to AI's practice request should be 'start', not 'one_off'
        assert result2["decision"] == "start", "Should detect user starting practice in response to AI"
        assert result2["confidence"] >= 0.7, "Should be confident about greeting practice"
    
    @pytest.mark.asyncio
    async def test_multi_turn_sales_practice_start(self, classifier):
        """Test multi-turn sales conversation start detection."""
        print("\n🧪 Testing multi-turn sales practice start...")
        
        # AI initiates sales practice
        ai_prompt = "lets practice a sales call. begin with an introduction, and try and sell me your product"
        result1 = await classifier.classify("assistant", ai_prompt, 0)
        
        print(f"AI sales prompt: {result1}")
        assert result1["decision"] in ["start", "continue"], "AI should initiate practice"
        
        # User starts sales pitch - provide context of AI's initiation
        context = [{"role": "assistant", "content": ai_prompt, "decision": result1["decision"]}]
        user_intro = "Hi, I am John from Spark Industries. I wanted to talk to you about our new product that could revolutionize your business operations."
        result2 = await classifier.classify("user", user_intro, 1, context=context)
        
        print(f"User sales intro: {result2}")
        # User responding to AI's practice request at turnCount=1 should be 'start'
        assert result2["decision"] == "start", "Should detect user starting practice in response to AI"
        assert result2["confidence"] >= 0.7, "Should be confident about sales practice"
    
    @pytest.mark.asyncio
    async def test_multi_turn_sales_practice_continue(self, classifier):
        """Test multi-turn sales conversation continuation."""
        print("\n🧪 Testing multi-turn sales practice continuation...")
        
        # Simulate ongoing sales conversation context
        context = "Previous: AI asked about product features, user explained benefits"
        
        # AI roleplay response with question
        ai_question = "(roleplaying) What kind of safety features are on your product? I'm concerned about liability."
        result = await classifier.classify("assistant", ai_question, 2)
        
        print(f"AI roleplay question: {result}")
        assert result["decision"] == "continue", "AI should continue roleplay"
        
        # User continues sales pitch
        user_continues = "Great question! Our product has triple-redundant safety systems, full compliance with industry standards, and a $2M liability insurance policy included."
        result = await classifier.classify("user", user_continues, 3)
        
        print(f"User continues pitch: {result}")
        assert result["decision"] == "continue", "Should detect continuation of practice"
        assert result["confidence"] >= 0.6, "Should be confident about ongoing practice"
    
    @pytest.mark.asyncio
    async def test_multi_turn_sales_practice_end(self, classifier):
        """Test multi-turn sales conversation end detection."""
        print("\n🧪 Testing multi-turn sales practice end...")
        
        # User closes sales pitch
        user_closes = "So that's why you should buy our product. What do you say? Are you ready to move forward with this investment?"
        result = await classifier.classify("user", user_closes, 4)
        
        print(f"User sales close: {result}")
        assert result["decision"] in ["end", "continue"], "Should detect closing attempt"
        
        # AI provides coaching feedback (end signal)
        ai_feedback = "Well done! Your clarity improved vastly since the last attempt. I would have 100% bought your product! Shall we practice your filler words next?"
        result = await classifier.classify("assistant", ai_feedback, 5)
        
        print(f"AI coaching feedback: {result}")
        assert result["decision"] == "end", "Should detect practice session end"
        assert result["confidence"] >= 0.8, "Should be very confident about session end"
    
    @pytest.mark.asyncio
    async def test_general_chitchat_filtering(self, classifier):
        """Test that general chitchat is filtered out (abstain)."""
        print("\n🧪 Testing general chitchat filtering...")
        
        chitchat_examples = [
            "How was your day today?",
            "What's the weather like?",
            "Did you see the game last night?",
            "I'm thinking about getting lunch.",
            "My cat is being weird today.",
            "Thanks for the help, see you later!",
        ]
        
        for i, message in enumerate(chitchat_examples):
            result = await classifier.classify("user", message, i)
            print(f"Chitchat '{message[:30]}...': {result['decision']} ({result['confidence']:.2f})")
            
            # Allow "end" for polite goodbyes, but should not be practice-related
            if "thanks" in message.lower() and "see you" in message.lower():
                assert result["decision"] in ["abstain", "end"], f"Goodbye should be abstain or end: {message}"
            else:
                assert result["decision"] == "abstain", f"Should abstain from chitchat: {message}"
            assert result["confidence"] >= 0.8, "Should be confident about non-practice content"
    
    @pytest.mark.asyncio
    async def test_filler_word_detection(self, classifier):
        """Test detection of speech with filler words."""
        print("\n🧪 Testing filler word detection...")
        
        filler_examples = [
            "this bed it the um most um comfiest bed in all of the earth!",
            "So um, our product is like, really good and um, you should buy it.",
            "Let me try this pitch: um, we have this amazing, uh, solution for you.",
            "I think that, uh, this is the best, um, approach we can take.",
        ]
        
        for i, message in enumerate(filler_examples):
            result = await classifier.classify("user", message, i)
            print(f"Filler words '{message[:40]}...': {result['decision']} ({result['confidence']:.2f})")
            
            # Filler words should be detected as speech practice (any decision except abstain)
            assert result["decision"] in ["one_off", "start", "end", "continue"], f"Should detect practice content: {message}"
            assert result["confidence"] >= 0.6, "Should be confident about speech practice"
    
    @pytest.mark.asyncio
    async def test_presentation_practice_detection(self, classifier):
        """Test detection of presentation and public speaking practice."""
        print("\n🧪 Testing presentation practice detection...")
        
        presentation_examples = [
            "I need to practice my presentation for tomorrow's board meeting.",
            "Let me rehearse this pitch one more time before the client call.",
            "Can you help me work on my public speaking skills?",
            "I want to practice explaining this concept more clearly.",
            "Let's run through my elevator pitch again.",
        ]
        
        for i, message in enumerate(presentation_examples):
            result = await classifier.classify("user", message, i)
            print(f"Presentation '{message[:40]}...': {result['decision']} ({result['confidence']:.2f})")
            
            # Presentation practice should be detected (start/continue based on context)
            assert result["decision"] in ["start", "one_off", "continue"], f"Should detect presentation practice: {message}"
            assert result["confidence"] >= 0.8, "Should be very confident about presentation practice"
    
    @pytest.mark.asyncio
    async def test_user_initiated_practice(self, classifier):
        """Test user-initiated practice scenarios."""
        print("\n🧪 Testing user-initiated practice...")
        
        user_initiated_examples = [
            "Right, let's do a sales practice. I'll start. Hello! My name is Sarah from TechCorp.",
            "I want to practice my elevator pitch. Here goes: Hi, I'm Mike and I help companies...",
        ]
        
        for message in user_initiated_examples:
            result = await classifier.classify("user", message, 0)
            print(f"User-initiated '{message[:40]}...': {result['decision']} ({result['confidence']:.2f})")
            # User-initiated practice can be 'start' or 'one_off' depending on content
            assert result["decision"] in ["start", "one_off"], f"Should detect user starting practice: {message}"
    
    @pytest.mark.asyncio
    async def test_context_awareness(self, classifier):
        """Test that classifier considers context from previous interactions."""
        print("\n🧪 Testing context awareness...")
        
        # Simulate a conversation context
        context_messages = [
            ("assistant", "Let's practice a customer service scenario.", 0),
            ("user", "Hello, how can I help you today?", 1),
            ("assistant", "I'm having trouble with my order.", 2),
            ("user", "I apologize for the inconvenience. Let me look into that for you.", 3),
        ]
        
        # Test each message in context with proper conversation history
        conversation_context = []
        for role, content, turn_count in context_messages:
            result = await classifier.classify(role, content, turn_count, context=conversation_context)
            print(f"Turn {turn_count} ({role}): {result['decision']} ({result['confidence']:.2f})")
            
            # Add to context for next message
            conversation_context.append({"role": role, "content": content, "decision": result["decision"]})
            
            if turn_count == 0:
                assert result["decision"] in ["start", "continue"], "AI should initiate practice"
            elif turn_count == 1:
                assert result["decision"] in ["start", "continue"], "User should start or continue practice"
            else:
                assert result["decision"] == "continue", "Should continue practice session"
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, classifier):
        """Test edge cases and boundary conditions."""
        print("\n🧪 Testing edge cases...")
        
        edge_cases = [
            ("", "empty_content"),  # Empty content
            ("um", "minimal_filler"),  # Just a filler word
            ("Hello.", "minimal_greeting"),  # Very short greeting
            ("Thank you very much for your time and consideration.", "polite_closing"),  # Polite but not practice
        ]
        
        for content, case_name in edge_cases:
            result = await classifier.classify("user", content, 0)
            print(f"Edge case '{case_name}': {result['decision']} ({result['confidence']:.2f})")
            
            # Should handle gracefully without errors
            assert result["decision"] in ["start", "continue", "end", "one_off", "abstain"]
            assert 0.0 <= result["confidence"] <= 1.0


# Helper function to run all tests
async def run_all_tests():
    """Run all classifier tests manually."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not set. Cannot run integration tests.")
        return
    
    print("🧪 Running Google LLM Classifier Integration Tests...")
    classifier = GoogleClassifierClient(model="gemini-1.5-flash")
    test_instance = TestGoogleClassifierIntegration()
    
    # Run each test
    tests = [
        test_instance.test_one_off_greeting_practice,
        test_instance.test_multi_turn_sales_practice_start,
        test_instance.test_multi_turn_sales_practice_continue,
        test_instance.test_multi_turn_sales_practice_end,
        test_instance.test_general_chitchat_filtering,
        test_instance.test_filler_word_detection,
        test_instance.test_presentation_practice_detection,
        test_instance.test_user_initiated_practice,
        test_instance.test_context_awareness,
        test_instance.test_edge_cases,
    ]  
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test.__name__}")
            await test(classifier)
            print(f"✅ PASSED: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test.__name__} - {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    # Run tests directly
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
