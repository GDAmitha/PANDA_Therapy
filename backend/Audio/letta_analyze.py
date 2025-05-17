from letta_client import Letta
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import from backend
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import database and letta agent manager
from database import Database
from letta_agent import LettaAgentManager

class TherapySessionAnalyzer:
    """Handles analysis of therapy sessions using the user's Letta agent"""
    
    def __init__(self):
        """Initialize the analyzer with Letta client and database"""
        self.db = Database()
        self.letta_manager = LettaAgentManager()
        logger.info("Therapy Session Analyzer initialized")
    
    def analyze_transcript(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Analyze a therapy session transcript for a specific user"""
        # Get the user and their agent
        user = self.db.get_user(user_id)
        if not user or not hasattr(user, 'letta_agent_id') or not user.letta_agent_id:
            logger.error(f"No Letta agent found for user {user_id}")
            return {"error": "No Letta agent found for this user"}
        
        # Get the transcript from the database
        transcript = self.db.get_transcript_by_session_id(user_id, session_id)
        if not transcript:
            logger.error(f"No transcript found for session {session_id}")
            return {"error": "Transcript not found"}
        
        # Process the transcript with the user's agent
        return self._process_transcript_with_agent(user.letta_agent_id, transcript)
    
    def analyze_transcript_data(self, user_id: str, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transcript data directly (without retrieving from DB)"""
        # Get the user and their agent
        user = self.db.get_user(user_id)
        if not user or not hasattr(user, 'letta_agent_id') or not user.letta_agent_id:
            logger.error(f"No Letta agent found for user {user_id}")
            return {"error": "No Letta agent found for this user"}
            
        # Process the transcript with the user's agent
        return self._process_transcript_with_agent(user.letta_agent_id, transcript_data)
    
    def _process_transcript_with_agent(self, agent_id: str, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a transcript with a specific Letta agent"""
        try:
            # Initialize the session analysis
            self.letta_manager.letta_client.agents.messages.create(
                agent_id=agent_id,
                messages=[{
                    "role": "system", 
                    "content": "You are a therapist analyzing a therapy session. You will be given speaker turns with text and emotional analysis. Follow your system instructions and analyze the session. I'll tell you when the session is complete."
                }]
            )
            
            # Process each utterance with emotion analysis
            emotion_analysis = transcript_data.get("emotion_analysis", [])
            if not emotion_analysis and "transcript" in transcript_data:
                # If no emotion analysis but transcript exists, use the transcript
                for entry in transcript_data["transcript"]:
                    self._process_utterance(agent_id, entry)
            else:
                # Process each entry in the emotion analysis
                for utterance in emotion_analysis:
                    self._process_utterance(agent_id, utterance)
            
            # Finalize the session analysis
            final_response = self.letta_manager.letta_client.agents.messages.create(
                agent_id=agent_id,
                messages=[{
                    "role": "system", 
                    "content": "The therapy session analysis is now complete. Please provide a summary of key insights from this session, including emotional patterns, significant topics, and therapeutic recommendations."
                }]
            )
            
            # Extract the summary from the response
            summary = ""
            for message in final_response.messages:
                # Handle different message types
                if hasattr(message, 'message_type') and message.message_type == "assistant_message":
                    summary = message.content
                # For backward compatibility with older Letta SDK versions
                elif hasattr(message, 'role') and message.role == "assistant":
                    summary = message.content
            
            # Return the analysis summary
            return {
                "status": "success",
                "analysis_summary": summary,
                "session_id": transcript_data.get("session_id", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transcript: {str(e)}")
            return {"error": f"Failed to analyze transcript: {str(e)}"}
    
    def _process_utterance(self, agent_id: str, utterance: Dict[str, Any]) -> None:
        """Process a single utterance from the transcript"""
        try:
            # Build the content based on available fields
            content_parts = []
            
            # Add speaker and text (these should always be present)
            speaker = utterance.get("speaker", "Unknown")
            text = utterance.get("text", "")
            content_parts.append(f"Speaker: {speaker}")
            content_parts.append(f"Text: {text}")
            
            # Add emotion analysis if available
            if "emotions" in utterance:
                emotions = utterance.get("emotions", ["neutral"])
                emotions_str = ", ".join(emotions)
                content_parts.append(f"Emotions: {emotions_str}")
            # For backward compatibility with older emotion format
            elif "predicted_wav_emotion" in utterance:
                wav_emotion = utterance.get("predicted_wav_emotion", "neutral")
                content_parts.append(f"Audio Emotion: {wav_emotion}")
                
                if "predicted_text_emotion" in utterance:
                    text_emotion = utterance.get("predicted_text_emotion", "neutral")
                    content_parts.append(f"Text Emotion: {text_emotion}")
            
            # Add confidence score if available
            if "confidence" in utterance:
                confidence = utterance.get("confidence", 0.0)
                content_parts.append(f"Confidence: {confidence:.2f}")
            
            # Send to the agent
            content = "\n".join(content_parts)
            self.letta_manager.letta_client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "system", "content": content}]
            )
            
        except Exception as e:
            logger.error(f"Error processing utterance: {str(e)}")

# For backward compatibility - simple function to analyze a file-based transcript
def analyze_therapy_session(transcript_path: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a therapy session from a transcript file"""
    # Load the transcript
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # If user_id is provided, use that user's agent
    analyzer = TherapySessionAnalyzer()
    if user_id:
        return analyzer.analyze_transcript_data(user_id, transcript_data)
    else:
        # Fallback to using the Letta client directly with a default agent
        client = Letta(token=os.getenv("LETTA_API_KEY"))
        try:
            # Use a default agent if available, otherwise retrieve the original one
            agent_id = os.getenv("DEFAULT_LETTA_AGENT_ID", "agent-e6f7f7c3-6063-4929-9e20-62716c7999d1")
            agent = client.agents.retrieve(agent_id=agent_id)
            
            # Process with default agent and return a simple result
            # This is just for backward compatibility
            client.agents.messages.create(
                agent_id=agent.id,
                messages=[{"role": "system", "content": "Analyzing therapy session"}]
            )
            return {"status": "success", "message": "Analyzed with default agent", "agent_id": agent.id}
        except Exception as e:
            logger.error(f"Error with default analysis: {str(e)}")
            return {"error": str(e)}
def print_message(message):
    if message.message_type == "reasoning_message":
        print("Reasoning:", message.reasoning)
    elif message.message_type == "assistant_message":
        print("Agent:", message.content)
    elif message.message_type == "tool_call_message":
        print("Tool Call:", message.tool_call.name)
        print("Arguments:", message.tool_call.arguments)
    elif message.message_type == "tool_return_message":
        print("Tool Return:", message.tool_return)
    elif message.message_type == "user_message":
        print("User Message:", message.content)

# Only run if executed directly
if __name__ == "__main__":
    import os
    # Use an example file if it exists
    sample_file = os.path.join(os.path.dirname(__file__), "audio_emo_transcript", "Mock Therapy Convo with Dimple_emotions.json")
    if os.path.exists(sample_file):
        print(f"Analyzing sample file: {sample_file}")
        analyze_therapy_session(sample_file)
    else:
        print("No sample file found for testing. Run with a specific file path.")
