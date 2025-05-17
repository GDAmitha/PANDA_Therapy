#!/usr/bin/env python3
"""
Add Transcript Entry Script

This script provides a helper tool to add new transcript entries to the database
with the proper format and metadata.
"""

import os
import sys
import json
import datetime
from pathlib import Path
import uuid
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import backend modules
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))


class TranscriptAdder:
    """
    Tool to add new transcript entries to the database with proper format
    """
    
    def __init__(self, transcript_path):
        """
        Initialize with the path to the transcript file
        
        Args:
            transcript_path: Path to the transcript JSON file
        """
        self.transcript_path = Path(transcript_path)
        self.transcripts = {}
        self.load_transcripts()
        
    def load_transcripts(self):
        """Load transcripts from the JSON file"""
        if self.transcript_path.exists():
            try:
                with open(self.transcript_path, 'r') as f:
                    self.transcripts = json.load(f)
                logger.info(f"Loaded {len(self.transcripts)} existing transcripts")
            except json.JSONDecodeError:
                logger.error("Failed to parse transcript file as JSON")
                raise
        else:
            logger.info(f"No existing transcript file found. Creating a new one.")
            self.transcripts = {}
    
    def save_transcripts(self):
        """Save the updated transcripts back to the JSON file"""
        # Create parent directories if they don't exist
        os.makedirs(self.transcript_path.parent, exist_ok=True)
            
        # Save the updated transcripts
        with open(self.transcript_path, 'w') as f:
            json.dump(self.transcripts, f, indent=2)
        logger.info(f"Saved transcripts to {self.transcript_path}")
    
    def add_transcript(self, therapist_name, client_name, transcript_entries, 
                      emotion_analysis=None, topic_summary=None, session_date=None):
        """
        Add a new transcript to the database
        
        Args:
            therapist_name: Name of the therapist
            client_name: Name of the client
            transcript_entries: List of {"speaker": name, "text": content} entries
            emotion_analysis: Optional list of emotion analysis results
            topic_summary: Optional topic summary
            session_date: Optional session date (ISO format)
        
        Returns:
            The session_id of the added transcript
        """
        # Generate a new session ID (next available integer)
        next_id = "1"
        if self.transcripts:
            try:
                next_id = str(max(int(k) for k in self.transcripts.keys()) + 1)
            except ValueError:
                # If keys can't be converted to integers, start from 1
                next_id = "1"
                
        # Current timestamp
        current_time = datetime.datetime.now().isoformat()
        
        # Use provided session date or current timestamp
        session_date = session_date or current_time
        
        # Generate a unique ID for the client
        client_id = str(uuid.uuid4())
        
        # Calculate estimated duration from timestamps if available
        duration = 50  # Default duration in minutes
        if transcript_entries and "end" in transcript_entries[-1] and "start" in transcript_entries[0]:
            try:
                duration = round((transcript_entries[-1]["end"] - transcript_entries[0]["start"]) / 60000)
                duration = max(1, duration)  # At least 1 minute
            except (KeyError, TypeError):
                pass
        
        # Generate a topic summary if not provided
        if not topic_summary and transcript_entries:
            client_entries = [e for e in transcript_entries if e.get("speaker") == client_name]
            if client_entries and len(client_entries[0]["text"]) > 20:
                topic_summary = client_entries[0]["text"][:100] + "..."
            else:
                topic_summary = "Therapy session"
        
        # Create the new transcript entry
        new_transcript = {
            "session_id": next_id,
            "therapist": {
                "id": f"therapist_{next_id}",
                "name": therapist_name
            },
            "client": {
                "id": client_id,
                "name": client_name
            },
            "session_info": {
                "date": session_date,
                "duration_minutes": duration,
                "topic_summary": topic_summary or "Therapy session",
                "session_number": int(next_id)
            },
            "transcript": transcript_entries,
            "emotion_analysis": emotion_analysis or [],
            "processed_at": current_time,
            "updated_at": current_time
        }
        
        # Add to transcripts and save
        self.transcripts[next_id] = new_transcript
        self.save_transcripts()
        
        logger.info(f"Added new transcript with session_id: {next_id}")
        return next_id


def main():
    parser = argparse.ArgumentParser(
        description="Add a new transcript entry to the database"
    )
    parser.add_argument(
        "--file", 
        default=str(Path(parent_dir) / "database" / "transcripts.json"),
        help="Path to the transcript JSON file"
    )
    parser.add_argument(
        "--therapist", 
        required=True,
        help="Name of the therapist"
    )
    parser.add_argument(
        "--client", 
        required=True,
        help="Name of the client"
    )
    parser.add_argument(
        "--transcript", 
        required=True,
        help="Path to a JSON file containing the transcript entries"
    )
    parser.add_argument(
        "--emotions",
        help="Path to a JSON file containing the emotion analysis"
    )
    parser.add_argument(
        "--topic",
        help="Brief summary of the session topic"
    )
    parser.add_argument(
        "--date",
        help="Session date in ISO format (YYYY-MM-DDTHH:MM:SS)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load transcript entries
        with open(args.transcript, 'r') as f:
            transcript_entries = json.load(f)
        
        # Load emotion analysis if provided
        emotion_analysis = None
        if args.emotions:
            try:
                with open(args.emotions, 'r') as f:
                    emotion_analysis = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load emotion analysis: {e}")
        
        # Add the transcript
        adder = TranscriptAdder(args.file)
        session_id = adder.add_transcript(
            args.therapist,
            args.client,
            transcript_entries,
            emotion_analysis,
            args.topic,
            args.date
        )
        
        print(f"✅ Added new transcript with session_id: {session_id}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
