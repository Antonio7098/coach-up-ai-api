#!/usr/bin/env python3
"""
Debugging script for diagnosing and fixing Convex interaction persistence issues.

This script provides tools to verify and diagnose issues with:
1. Convex URL configuration
2. Session validation
3. Interaction persistence
4. Connection timeouts
"""

import os
import sys
import json
import logging
import argparse
import uuid
import hashlib
import asyncio
import httpx
from typing import Dict, Any, Optional, Tuple

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def check_convex_url() -> Optional[str]:
    """Check if the Convex URL is properly configured."""
    base = (os.getenv("CONVEX_URL") or "").strip()
    if not base:
        logger.error("CONVEX_URL is not set in environment variables")
        return None
    
    logger.info(f"CONVEX_URL is set to: {base}")
    return base

async def test_convex_connectivity(base_url: str) -> bool:
    """Test basic connectivity to Convex."""
    if not base_url:
        return False
    
    url = base_url.rstrip("/") + "/api/query"
    payload = {
        "path": "functions/sessions:listRecentSessions",
        "args": {"limit": 1},
        "format": "json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url, 
                json=payload, 
                headers={"Content-Type": "application/json"}
            )
            
            if resp.status_code >= 400:
                logger.error(f"Connectivity test failed: HTTP {resp.status_code}")
                logger.error(f"Response: {resp.text}")
                return False
            
            data = resp.json()
            logger.info(f"Connectivity test successful: {data.get('status', 'unknown')}")
            return True
    except Exception as e:
        logger.error(f"Connectivity test failed with exception: {e}")
        return False

async def verify_session(base_url: str, session_id: str) -> Optional[Dict[str, Any]]:
    """Verify if a session exists and return its data."""
    if not base_url or not session_id:
        return None
    
    url = base_url.rstrip("/") + "/api/query"
    payload = {
        "path": "functions/sessions:getBySessionId",
        "args": {"sessionId": session_id},
        "format": "json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url, 
                json=payload, 
                headers={"Content-Type": "application/json"}
            )
            
            if resp.status_code >= 400:
                logger.error(f"Session verification failed: HTTP {resp.status_code}")
                return None
            
            data = resp.json()
            if data.get("status") == "error":
                logger.error(f"Session verification failed: {data.get('error', 'unknown error')}")
                return None
            
            session_doc = data.get("value")
            if session_doc:
                logger.info(f"Session verified: {session_id}")
                return session_doc
            else:
                logger.error(f"Session not found: {session_id}")
                return None
    except Exception as e:
        logger.error(f"Session verification failed with exception: {e}")
        return None

async def list_interactions(base_url: str, session_id: str) -> Tuple[int, int]:
    """List interactions for a session and count user/assistant messages."""
    if not base_url or not session_id:
        return (0, 0)
    
    url = base_url.rstrip("/") + "/api/query"
    payload = {
        "path": "functions/interactions:listBySession",
        "args": {"sessionId": session_id, "limit": 100},
        "format": "json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url, 
                json=payload, 
                headers={"Content-Type": "application/json"}
            )
            
            if resp.status_code >= 400:
                logger.error(f"List interactions failed: HTTP {resp.status_code}")
                return (0, 0)
            
            data = resp.json()
            if data.get("status") == "error":
                logger.error(f"List interactions failed: {data.get('error', 'unknown error')}")
                return (0, 0)
            
            interactions = data.get("value", [])
            user_count = sum(1 for i in interactions if i.get("role") == "user")
            assistant_count = sum(1 for i in interactions if i.get("role") == "assistant")
            
            logger.info(f"Session {session_id} has {len(interactions)} interactions")
            logger.info(f"- User messages: {user_count}")
            logger.info(f"- Assistant messages: {assistant_count}")
            
            return (user_count, assistant_count)
    except Exception as e:
        logger.error(f"List interactions failed with exception: {e}")
        return (0, 0)

async def create_test_interaction(base_url: str, session_id: str, role: str = "assistant") -> bool:
    """Create a test interaction to verify persistence."""
    if not base_url or not session_id:
        return False
    
    message_id = str(uuid.uuid4())
    content = f"Test message created at {uuid.uuid4()}"
    content_hash = hashlib.sha256(f"{role}|{content}".encode("utf-8")).hexdigest()
    ts_ms = int(asyncio.get_event_loop().time() * 1000)
    
    url = base_url.rstrip("/") + "/api/mutation"
    payload = {
        "path": "functions/interactions:appendInteraction",
        "args": {
            "sessionId": session_id,
            "messageId": message_id,
            "role": role,
            "contentHash": content_hash,
            "text": content,
            "ts": ts_ms
        },
        "format": "json"
    }
    
    try:
        logger.info(f"Creating test {role} interaction for session {session_id}")
        logger.info(f"Payload: {json.dumps(payload, indent=2)}")
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                url, 
                json=payload, 
                headers={"Content-Type": "application/json"}
            )
            
            if resp.status_code >= 400:
                logger.error(f"Create interaction failed: HTTP {resp.status_code}")
                logger.error(f"Response: {resp.text}")
                return False
            
            try:
                data = resp.json()
                logger.info(f"Create interaction response: {json.dumps(data, indent=2)}")
                return data.get("status") == "success"
            except Exception as e:
                logger.error(f"Failed to parse response JSON: {e}")
                logger.info(f"Raw response: {resp.text[:500]}")
                return False
    except Exception as e:
        logger.error(f"Create interaction failed with exception: {e}")
        return False

async def main(args):
    # Check and validate Convex URL
    base_url = await check_convex_url()
    if not base_url:
        logger.error("CONVEX_URL is not configured. Please set it in your .env file")
        return 1
    
    # Test basic Convex connectivity
    if not await test_convex_connectivity(base_url):
        logger.error("Failed to connect to Convex. Check your URL and network connectivity")
        return 1
    
    # Verify session exists
    if args.session_id:
        session_doc = await verify_session(base_url, args.session_id)
        if not session_doc:
            logger.error(f"Session {args.session_id} does not exist or cannot be retrieved")
            if args.create_session:
                logger.info(f"Creating session manually is not supported by this script")
            return 1
        
        # List existing interactions
        user_count, assistant_count = await list_interactions(base_url, args.session_id)
        
        # Create test interactions if requested
        if args.create_test:
            logger.info("Creating test interactions...")
            
            # Create user message if requested
            if args.create_user:
                logger.info("Creating test user message...")
                if await create_test_interaction(base_url, args.session_id, "user"):
                    logger.info("Successfully created test user message")
                else:
                    logger.error("Failed to create test user message")
            
            # Create assistant message
            logger.info("Creating test assistant message...")
            if await create_test_interaction(base_url, args.session_id, "assistant"):
                logger.info("Successfully created test assistant message")
            else:
                logger.error("Failed to create test assistant message")
            
            # Verify the new interactions were created
            new_user_count, new_assistant_count = await list_interactions(base_url, args.session_id)
            logger.info(f"User messages: {user_count} -> {new_user_count}")
            logger.info(f"Assistant messages: {assistant_count} -> {new_assistant_count}")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug and fix Convex interaction persistence issues")
    parser.add_argument("--session-id", help="Session ID to check or use for tests")
    parser.add_argument("--create-test", action="store_true", help="Create test interactions")
    parser.add_argument("--create-user", action="store_true", help="Create a test user interaction")
    parser.add_argument("--create-session", action="store_true", help="Create a session if it doesn't exist")
    
    args = parser.parse_args()
    
    if not args.session_id and (args.create_test or args.create_session):
        parser.error("--session-id is required when using --create-test or --create-session")
    
    asyncio.run(main(args))
