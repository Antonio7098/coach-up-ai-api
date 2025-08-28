#!/usr/bin/env python3
"""
Google Assessment Integration Tests
Tests the Google Gemini API for speech coaching assessments with skill context.
"""

import os
import asyncio
import pytest
from typing import Dict, Any

# Import the Google assess client
from app.providers.google import GoogleAssessClient


class TestGoogleAssessIntegration:
    """Integration tests for Google assessment client."""

    def get_assess_client(self):
        """Create a Google assess client for testing."""
        # Use environment variable or fallback model
        model = os.getenv("AI_ASSESS_MODEL", "gemini-1.5-flash")
        return GoogleAssessClient(model=model)

    async def test_basic_assessment(self):
        """Test basic assessment functionality."""
        print("\n🧪 Testing basic Google assessment...")
        
        assess_client = self.get_assess_client()
        
        # Sample transcript with filler words and clarity issues
        transcript = [
            {"role": "user", "content": "Um, so like, I think our product is, uh, really good and stuff."},
            {"role": "user", "content": "It's like, you know, the best thing ever, um, for businesses."},
        ]
        
        result = await assess_client.assess(transcript, rubric_version="v2")
        
        print(f"Basic assessment result: {result}")
        
        # Verify structure
        assert "rubricVersion" in result
        assert "level" in result
        assert "highlights" in result
        assert "recommendations" in result
        assert "rubricKeyPoints" in result
        assert "meta" in result
        
        assert result["rubricVersion"] == "v2"
        assert result["meta"]["provider"] == "google"
        assert isinstance(result["level"], int)
        assert 1 <= result["level"] <= 10
        
        print("✅ PASSED: test_basic_assessment")

    async def test_skill_focused_assessment(self):
        """Test assessment with specific skill context."""
        print("\n🧪 Testing skill-focused assessment...")
        
        assess_client = self.get_assess_client()
        
        # Sample transcript with filler word issues
        transcript = [
            {"role": "user", "content": "Um, so like, I think our product is, uh, really good and stuff."},
            {"role": "user", "content": "It's like, you know, the best thing ever, um, for businesses and, uh, customers."},
        ]
        
        # Mock skill with detailed context
        skill = {
            "id": "filler_words",
            "name": "Filler Words",
            "category": "delivery",
            "currentLevel": 3,
            "criteria": {
                "1": "Excessive filler words (>10 per minute)",
                "3": "Moderate filler words (5-10 per minute)",
                "5": "Few filler words (2-5 per minute)",
                "8": "Minimal filler words (<2 per minute)",
                "10": "No noticeable filler words"
            },
            "rubric": {
                "frequency": "Count of um, uh, like, you know per minute",
                "awareness": "Recognition and self-correction of fillers",
                "fluency": "Overall speech flow and confidence"
            },
            "examples": {
                "3": "Um, so like, I think our product is, uh, really good",
                "8": "I believe our product offers significant value to customers"
            }
        }
        
        result = await assess_client.assess(
            transcript, 
            rubric_version="v2", 
            skill=skill,
            request_id="test-skill-123"
        )
        
        print(f"Skill-focused assessment: {result}")
        
        # Verify skill context is included
        assert result["meta"]["skill"]["id"] == "filler_words"
        assert result["meta"]["skill"]["currentLevel"] == 3
        assert result["meta"]["requestId"] == "test-skill-123"
        
        # Should have specific feedback related to filler words
        assert isinstance(result["level"], int)
        assert 1 <= result["level"] <= 10
        assert len(result["highlights"]) > 0
        assert len(result["recommendations"]) > 0
        assert len(result["rubricKeyPoints"]) > 0
        
        print("✅ PASSED: test_skill_focused_assessment")

    async def test_clarity_assessment(self):
        """Test assessment for clarity skill."""
        print("\n🧪 Testing clarity skill assessment...")
        
        assess_client = self.get_assess_client()
        
        # Sample transcript with clarity issues
        transcript = [
            {"role": "user", "content": "So basically what I'm trying to say is that the thing we're doing is good."},
            {"role": "user", "content": "The solution we have addresses the problem that exists in the market space."},
        ]
        
        skill = {
            "id": "clarity",
            "name": "Clarity",
            "category": "content",
            "currentLevel": 4,
            "criteria": {
                "1": "Very unclear, confusing message",
                "4": "Somewhat clear with some vague statements",
                "7": "Clear and easy to understand",
                "10": "Crystal clear, precise communication"
            }
        }
        
        result = await assess_client.assess(transcript, skill=skill)
        
        print(f"Clarity assessment: {result}")
        
        assert result["meta"]["skill"]["id"] == "clarity"
        assert isinstance(result["level"], int)
        assert 1 <= result["level"] <= 10
        assert len(result["recommendations"]) >= 3  # Should provide actionable tips
        
        print("✅ PASSED: test_clarity_assessment")

    async def test_energy_assessment(self):
        """Test assessment for energy/enthusiasm skill."""
        print("\n🧪 Testing energy skill assessment...")
        
        assess_client = self.get_assess_client()
        
        # Sample transcript with low energy
        transcript = [
            {"role": "user", "content": "I guess our product is okay. It might help some people."},
            {"role": "user", "content": "We have some features that could be useful, I suppose."},
        ]
        
        skill = {
            "id": "energy",
            "name": "Energy & Enthusiasm",
            "category": "delivery",
            "currentLevel": 2,
            "criteria": {
                "1": "Monotone, no enthusiasm",
                "2": "Low energy, tentative language",
                "5": "Moderate energy and confidence",
                "8": "High energy, engaging delivery",
                "10": "Exceptional enthusiasm and passion"
            }
        }
        
        result = await assess_client.assess(transcript, skill=skill)
        
        print(f"Energy assessment: {result}")
        
        assert result["meta"]["skill"]["id"] == "energy"
        assert isinstance(result["level"], int)
        assert 1 <= result["level"] <= 10
        # Should identify low energy and provide recommendations
        assert len(result["recommendations"]) >= 3
        
        print("✅ PASSED: test_energy_assessment")

    async def test_mixed_role_filtering(self):
        """Test that assessment focuses only on user messages."""
        print("\n🧪 Testing role filtering...")
        
        assess_client = self.get_assess_client()
        
        # Mixed transcript with assistant and user messages
        transcript = [
            {"role": "assistant", "content": "Please give me your elevator pitch."},
            {"role": "user", "content": "Um, so our company makes, like, really good software."},
            {"role": "assistant", "content": "Can you be more specific?"},
            {"role": "user", "content": "We help businesses, uh, manage their data better."},
        ]
        
        result = await assess_client.assess(transcript, target_role="user")
        
        print(f"Role filtering result: {result}")
        
        # Should only assess user messages
        assert result["meta"]["targetRole"] == "user"
        assert len(result["highlights"]) > 0
        
        print("✅ PASSED: test_mixed_role_filtering")

    async def test_error_handling(self):
        """Test error handling with invalid input."""
        print("\n🧪 Testing error handling...")
        
        assess_client = self.get_assess_client()
        
        # Test with empty transcript
        result = await assess_client.assess([])
        
        print(f"Empty transcript result: {result}")
        
        # Should return valid structure even with no content
        assert "rubricVersion" in result
        assert "level" in result
        assert "highlights" in result
        assert "recommendations" in result
        assert "rubricKeyPoints" in result
        assert "meta" in result
        
        print("✅ PASSED: test_error_handling")


async def main():
    """Run all Google assessment integration tests."""
    print("🧪 Running Google Assessment Integration Tests...")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found. Skipping integration tests.")
        return
    
    test_instance = TestGoogleAssessIntegration()
    
    tests = [
        test_instance.test_basic_assessment(),
        test_instance.test_skill_focused_assessment(),
        test_instance.test_clarity_assessment(),
        test_instance.test_energy_assessment(),
        test_instance.test_mixed_role_filtering(),
        test_instance.test_error_handling(),
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")


if __name__ == "__main__":
    asyncio.run(main())
