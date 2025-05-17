#!/usr/bin/env python3
"""
Update and Vectorize Natey's Transcript

This script:
1. Updates Natey's transcript in transcripts.json to the new format
2. Vectorizes the transcript for improved memory retrieval
"""

import os
import sys
import json
import logging
import datetime
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import backend modules
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

# Import transcript memory system
try:
    from backend.transcript_memory import TherapyTranscriptMemory
except ImportError as e:
    logger.error(f"Error importing transcript_memory: {e}")
    sys.exit(1)

TRANSCRIPTS_PATH = os.path.join(parent_dir, "database", "transcripts.json")


def update_natey_transcript():
    """Update Natey's transcript with the new structured format"""
    try:
        # Load the current transcripts
        with open(TRANSCRIPTS_PATH, 'r') as f:
            transcripts = json.load(f)
        
        # Make sure we have a backup
        backup_path = TRANSCRIPTS_PATH + ".backup"
        with open(backup_path, 'w') as f:
            json.dump(transcripts, f, indent=2)
        logger.info(f"Created backup at {backup_path}")
        
        # Get Natey's transcript (session 8)
        if "8" not in transcripts:
            logger.error("Session 8 (Natey's transcript) not found")
            return False
            
        natey_transcript = transcripts["8"]
        
        # Create the updated transcript with structured format
        session_id = "8"
        
        # Extract transcript content
        transcript_content = natey_transcript.get("transcript", [])
        emotion_analysis = natey_transcript.get("emotion_analysis", [])
        
        # Generate a unique ID for Natey
        natey_id = str(uuid.uuid4())
        
        # Calculate session duration based on timestamps
        duration_minutes = 30  # Default
        if transcript_content and len(transcript_content) > 1:
            try:
                first_ts = transcript_content[0].get("start", 0)
                last_ts = transcript_content[-1].get("end", 0)
                if first_ts and last_ts:
                    duration_ms = last_ts - first_ts
                    duration_minutes = max(1, round(duration_ms / 60000))  # Convert ms to minutes
            except Exception as e:
                logger.warning(f"Could not calculate session duration: {e}")
        
        # Generate topic summary from Natey's first substantial response
        topic_summary = "Therapy session about impulsive behavior"
        for entry in transcript_content:
            if entry.get("speaker") == "natey" and len(entry.get("text", "")) > 20:
                text = entry.get("text", "")
                topic_summary = text[:100] + "..." if len(text) > 100 else text
                break
        
        # Create the structured transcript
        updated_transcript = {
            "session_id": session_id,
            "therapist": {
                "id": f"therapist_{session_id}",
                "name": "Dr. YM"
            },
            "client": {
                "id": natey_id,
                "name": "natey"
            },
            "session_info": {
                "date": natey_transcript.get("created_at", datetime.datetime.now().isoformat()),
                "duration_minutes": duration_minutes,
                "topic_summary": topic_summary,
                "session_number": int(session_id)
            },
            "transcript": transcript_content,
            "emotion_analysis": emotion_analysis,
            "processed_at": natey_transcript.get("processed_at", datetime.datetime.now().timestamp()),
            "updated_at": datetime.datetime.now().isoformat()
        }
        
        # Update the transcripts dictionary
        transcripts[session_id] = updated_transcript
        
        # Save the updated transcripts
        with open(TRANSCRIPTS_PATH, 'w') as f:
            json.dump(transcripts, f, indent=2)
        
        logger.info(f"Successfully updated Natey's transcript (session {session_id})")
        return True, natey_id
        
    except Exception as e:
        logger.error(f"Error updating Natey's transcript: {e}")
        return False, None


def vectorize_transcript(user_id):
    """
    Vectorize Natey's transcript using the transcript memory system
    
    Args:
        user_id: Natey's user ID for transcript retrieval
    """
    try:
        logger.info(f"Initializing memory system for user {user_id}")
        memory = TherapyTranscriptMemory(user_id)
        
        # Force reload to update with the new transcript format
        success = memory.create_memory_index(force_reload=True)
        
        if success:
            logger.info(f"Successfully vectorized transcript for user {user_id}")
            return True
        else:
            logger.error(f"Failed to vectorize transcript for user {user_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error vectorizing transcript: {e}")
        return False


def test_memory_retrieval(user_id, query):
    """
    Test retrieval of memories from the vectorized transcript
    
    Args:
        user_id: User ID for memory retrieval
        query: Query to test memory retrieval
    """
    try:
        memory = TherapyTranscriptMemory(user_id)
        
        # Extract memories for the query
        memories = memory.extract_memories(query, max_results=3)
        
        if not memories:
            logger.warning("No memories retrieved for the query")
            return
            
        logger.info(f"Retrieved {len(memories)} memories for query: '{query}'")
        
        # Format memories for display
        formatted = memory._format_memory_for_letta(memories)
        print("\n" + "="*80)
        print("RETRIEVED MEMORIES:")
        print("="*80)
        print(formatted)
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error testing memory retrieval: {e}")


def main():
    # Update Natey's transcript
    success, user_id = update_natey_transcript()
    if not success:
        logger.error("Failed to update Natey's transcript")
        sys.exit(1)
    
    # Vectorize the transcript
    success = vectorize_transcript(user_id)
    if not success:
        logger.error("Failed to vectorize the transcript")
        sys.exit(1)
    
    # Test memory retrieval
    test_memory_retrieval(user_id, "Why did Natey jump into the lake?")
    test_memory_retrieval(user_id, "What emotions was Natey feeling?")
    test_memory_retrieval(user_id, "How was Natey saved?")
    
    logger.info("Script completed successfully")


if __name__ == "__main__":
    main()
