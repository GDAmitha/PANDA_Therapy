#!/usr/bin/env python3
"""
Transcript Format Updater

This script updates the format of therapy transcripts in the database,
adding proper metadata and organization to clearly identify participants
and session details.
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


class TranscriptFormatter:
    """
    Updates therapy transcript formats with improved metadata and organization.
    """
    
    def __init__(self, transcript_path):
        """
        Initialize the formatter with the path to the transcript file
        
        Args:
            transcript_path: Path to the transcript JSON file
        """
        self.transcript_path = Path(transcript_path)
        self.transcripts = {}
        self.load_transcripts()
        
    def load_transcripts(self):
        """Load transcripts from the JSON file"""
        if not self.transcript_path.exists():
            logger.error(f"Transcript file not found: {self.transcript_path}")
            raise FileNotFoundError(f"Transcript file not found: {self.transcript_path}")
        
        try:
            with open(self.transcript_path, 'r') as f:
                self.transcripts = json.load(f)
            logger.info(f"Loaded {len(self.transcripts)} transcripts from {self.transcript_path}")
        except json.JSONDecodeError:
            logger.error("Failed to parse transcript file as JSON")
            raise
            
    def save_transcripts(self):
        """Save the updated transcripts back to the JSON file"""
        # Create a backup of the original file
        backup_path = str(self.transcript_path) + ".backup"
        with open(backup_path, 'w') as f:
            json.dump(self.transcripts, f, indent=2)
        logger.info(f"Created backup at {backup_path}")
            
        # Save the updated transcripts
        with open(self.transcript_path, 'w') as f:
            json.dump(self.transcripts, f, indent=2)
        logger.info(f"Saved updated transcripts to {self.transcript_path}")
    
    def update_format(self):
        """Update the format of all transcripts"""
        updated_transcripts = {}
        
        for session_id, transcript_data in self.transcripts.items():
            logger.info(f"Processing transcript {session_id}")
            
            # Extract transcript content
            transcript = transcript_data.get("transcript", [])
            
            if not transcript:
                logger.warning(f"Skipping empty transcript {session_id}")
                continue
            
            # Identify therapist and client from speakers
            speakers = set(entry.get("speaker") for entry in transcript)
            
            # Get information needed for the updated format
            therapist_name = self._determine_therapist(speakers)
            client_name = self._determine_client(speakers, therapist_name)
            
            # Current timestamp for the updated record
            current_time = datetime.datetime.now().isoformat()
            
            # Format session date (use creation date if available, otherwise current date)
            session_date = transcript_data.get("created_at", current_time)
            
            # Create a UUID for the session if not present
            user_id = transcript_data.get("user_id", str(uuid.uuid4()))
            
            # Create updated transcript record
            updated_transcript = {
                "session_id": session_id,
                "therapist": {
                    "id": f"therapist_{session_id}",
                    "name": therapist_name
                },
                "client": {
                    "id": user_id,
                    "name": client_name
                },
                "session_info": {
                    "date": session_date,
                    "duration_minutes": self._estimate_session_duration(transcript),
                    "topic_summary": self._generate_topic_summary(transcript),
                    "session_number": int(session_id)  # Assuming sequential numbering
                },
                "transcript": transcript,
                "emotion_analysis": transcript_data.get("emotion_analysis", []),
                "processed_at": transcript_data.get("processed_at", current_time),
                "updated_at": current_time
            }
            
            updated_transcripts[session_id] = updated_transcript
            
        self.transcripts = updated_transcripts
        return len(updated_transcripts)
    
    def _determine_therapist(self, speakers):
        """
        Determine which speaker is likely the therapist
        
        Simple heuristic: Looks for speaker names containing 'Dr', 'Therapist', etc.
        """
        therapist_indicators = ["dr", "doctor", "therapist", "counselor", "psychologist"]
        
        for speaker in speakers:
            speaker_lower = speaker.lower()
            if any(indicator in speaker_lower for indicator in therapist_indicators):
                return speaker
                
        # If no clear therapist indicator, assume the first speaker is the therapist
        return list(speakers)[0] if speakers else "Therapist"
    
    def _determine_client(self, speakers, therapist_name):
        """
        Determine which speaker is the client (anyone who's not the therapist)
        """
        for speaker in speakers:
            if speaker != therapist_name:
                return speaker
                
        # Fallback if only one speaker is found
        return "Client"
    
    def _estimate_session_duration(self, transcript):
        """
        Estimate session duration in minutes based on timestamps if available
        """
        try:
            # Check if timestamps are available
            if transcript and "end" in transcript[-1] and "start" in transcript[0]:
                last_ts = transcript[-1]["end"] 
                first_ts = transcript[0]["start"]
                # Convert milliseconds to minutes
                duration_mins = round((last_ts - first_ts) / 60000)
                return max(1, duration_mins)  # At least 1 minute
        except (KeyError, TypeError, IndexError):
            pass
            
        # Default duration if timestamps not available
        return 50  # Typical therapy session
    
    def _generate_topic_summary(self, transcript):
        """
        Generate a simple topic summary based on the first few exchanges
        """
        # For now, just extract from the first client response if it's long enough
        client_responses = [entry["text"] for entry in transcript 
                           if entry.get("speaker") != self._determine_therapist(
                               set(entry.get("speaker") for entry in transcript))]
        
        if client_responses and len(client_responses[0]) > 20:
            # Use the first 100 characters as a summary
            return client_responses[0][:100] + "..."
        
        return "Therapy session"


def create_transcript_template():
    """Return a template for a new transcript entry"""
    template = {
        "session_id": "SESSION_ID",
        "therapist": {
            "id": "THERAPIST_ID",
            "name": "Therapist Name"
        },
        "client": {
            "id": "CLIENT_ID",
            "name": "Client Name"
        },
        "session_info": {
            "date": "YYYY-MM-DDTHH:MM:SS",
            "duration_minutes": 50,
            "topic_summary": "Brief summary of session topic",
            "session_number": 1
        },
        "transcript": [
            {
                "speaker": "Therapist Name",
                "text": "Hello, how are you today?",
                "start": 0,  # Timestamp in milliseconds if available
                "end": 3000  # Timestamp in milliseconds if available
            },
            {
                "speaker": "Client Name",
                "text": "I'm doing well, thank you.",
                "start": 3500,
                "end": 5500
            }
        ],
        "emotion_analysis": [
            {
                "speaker": "Client Name",
                "text": "I'm doing well, thank you.",
                "emotions": ["neutral", "joy"],
                "confidence": 0.8
            }
        ],
        "processed_at": "YYYY-MM-DDTHH:MM:SS",
        "updated_at": "YYYY-MM-DDTHH:MM:SS"
    }
    
    return json.dumps(template, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Update therapy transcript format with improved metadata"
    )
    parser.add_argument(
        "--file", 
        default=str(Path(parent_dir) / "database" / "transcripts.json"),
        help="Path to the transcript JSON file"
    )
    parser.add_argument(
        "--template", 
        action="store_true",
        help="Output a template for new transcript entries"
    )
    args = parser.parse_args()
    
    if args.template:
        print(create_transcript_template())
        return
    
    try:
        formatter = TranscriptFormatter(args.file)
        count = formatter.update_format()
        formatter.save_transcripts()
        logger.info(f"Successfully updated {count} transcripts")
        print(f"✅ Updated {count} transcripts with improved metadata")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
