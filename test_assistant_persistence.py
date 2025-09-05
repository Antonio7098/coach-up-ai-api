#!/usr/bin/env python3
"""
Test script to directly test assistant message persistence.
"""

import sys
sys.path.append('/home/antonio/programming/coach-up/coach-up-ai-api')

import asyncio
import uuid
import time
from dotenv import load_dotenv
load_dotenv()

# Import the main module functions
from app.main import _persist_interaction_if_configured, logger

async def test_assistant_persistence():
    """Test assistant message persistence directly."""
    
    print("Testing assistant message persistence...")
    
    # Use the session ID from the issue
    session_id = "149aa587-94db-4196-a451-78ec5ea1b0b6"
    message_id = str(uuid.uuid4())
    role = "assistant"
    content = "Hello! I'm a speech coach here to help you improve your communication skills."
    ts_ms = int(time.time() * 1000)
    group_id = None  # Optional
    
    print(f"Session ID: {session_id}")
    print(f"Message ID: {message_id}")
    print(f"Role: {role}")
    print(f"Content length: {len(content)}")
    print(f"Timestamp: {ts_ms}")
    print()
    
    # Call the persistence function
    try:
        await _persist_interaction_if_configured(
            session_id=session_id,
            group_id=group_id,
            message_id=message_id,
            role=role,
            content=content,
            ts_ms=ts_ms,
        )
        print("✅ Assistant message persistence completed without errors")
    except Exception as e:
        print(f"❌ Assistant message persistence failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_assistant_persistence())
