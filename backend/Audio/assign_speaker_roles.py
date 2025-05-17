# Main entry point for clients interaction with chatbot
import os
import json
import logging
from typing import Dict, Any, Union
from pathlib import Path
from letta_client import Letta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Letta client
client = Letta(token=os.getenv("LETTA_API_KEY"))
agent_assign_roles = client.agents.retrieve(agent_id="agent-3f300241-137e-43bc-bea8-e7ee3185e973")


def assign_speaker_roles(transcript_path: Union[str, Dict[str, Any]], therapist_name: str = "Therapist", patient_name: str = "Client") -> str:
    """
    Assign speaker roles to a transcript and save the result to a new JSON file.
    
    Args:
        transcript_path: Path to the JSON transcript file or transcript data dictionary
        therapist_name: Name of the therapist to assign
        patient_name: Name of the patient to assign
        
    Returns:
        Path to the new transcript file with assigned roles
    """
    # Load transcript data
    try:
        if isinstance(transcript_path, dict):
            # If transcript_path is actually a dictionary of transcript data
            transcript = transcript_path
            # Generate a filename since we don't have a real path
            from datetime import datetime
            current_dir = os.path.dirname(os.path.abspath(__file__))  # Get Audio folder path
            base_dir = os.path.join(current_dir, "speaker_assigned_transcript")
            os.makedirs(base_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"transcript_{timestamp}_assigned.json"
            output_path = os.path.join(base_dir, output_filename)
        else:
            # If transcript_path is a string path to a file
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            # Create output directory if it doesn't exist
            current_dir = os.path.dirname(os.path.abspath(__file__))  # Get Audio folder path
            base_dir = os.path.join(current_dir, "speaker_assigned_transcript")
            os.makedirs(base_dir, exist_ok=True)
            # Generate output filename
            filename = os.path.basename(transcript_path).split('.')[0]
            output_path = os.path.join(base_dir, f"{filename}_assigned.json")
    except Exception as e:
        logger.error(f"Error loading transcript: {str(e)}")
        # Return original path/data if we can't process
        return transcript_path
    
    # Get first 6 utterances for role analysis
    first_six = transcript["utterances"][:6]
    
    # Create a conversation string for GPT
    conversation = ""
    for utt in first_six:
        speaker = utt["speaker"]
        text = utt["text"]
        conversation += f"Speaker {speaker}: {text}\n"
    
    try:
        # Call Letta agent to assign speaker roles
        response = client.agents.messages.create(
            agent_id=agent_assign_roles.id,
            messages=[
                {
                    "role": "user",
                    "content": conversation
                }
            ]
        )
        
        # Parse the response
        therapist_speaker = None
        client_speaker = None
        
        # Try multiple parsing strategies for Letta response
        for message in response.messages:
            if message.message_type == "assistant_message":
                result = message.content.strip()
                logger.info(f"Raw Letta response for speaker assignment: {result}")
                
                # Strategy 1: Look for key=value pattern
                try:
                    # Normalize various terms
                    normalized_result = result.lower()
                    normalized_result = normalized_result.replace('therapist', 'therapist')
                    normalized_result = normalized_result.replace('counselor', 'therapist')
                    normalized_result = normalized_result.replace('doctor', 'therapist')
                    normalized_result = normalized_result.replace('psychologist', 'therapist')
                    normalized_result = normalized_result.replace('client', 'client')
                    normalized_result = normalized_result.replace('patient', 'client')
                    
                    # Split by common delimiters
                    for delimiter in [',', ';', '\n']:
                        if delimiter in normalized_result:
                            parts = normalized_result.split(delimiter)
                            for part in parts:
                                if '=' in part or ':' in part:
                                    # Handle both key=value and key: value formats
                                    if '=' in part:
                                        role, spkr = part.split('=')
                                    else:
                                        role, spkr = part.split(':')
                                        
                                    role = role.strip().lower()
                                    spkr = spkr.strip().upper()
                                    
                                    if 'therapist' in role:
                                        therapist_speaker = spkr
                                    elif 'client' in role:
                                        client_speaker = spkr
                except Exception as e:
                    logger.warning(f"Strategy 1 failed: {str(e)}")
                
                # Strategy 2: Look for specific phrases
                if therapist_speaker is None or client_speaker is None:
                    try:
                        lower_result = result.lower()
                        sentences = lower_result.split('.')
                        
                        for sentence in sentences:
                            # Look for phrases like "Speaker A is the therapist"
                            if 'is the therapist' in sentence or 'as the therapist' in sentence:
                                words = sentence.split()
                                for i, word in enumerate(words):
                                    if word.upper() in ["A", "B", "C", "D", "SPEAKER_A", "SPEAKER_B"] and i > 0:
                                        therapist_speaker = word.upper().replace('SPEAKER_', '')
                                        break
                            
                            # Look for phrases like "Speaker B is the client"
                            if 'is the client' in sentence or 'as the client' in sentence or 'is the patient' in sentence:
                                words = sentence.split()
                                for i, word in enumerate(words):
                                    if word.upper() in ["A", "B", "C", "D", "SPEAKER_A", "SPEAKER_B"] and i > 0:
                                        client_speaker = word.upper().replace('SPEAKER_', '')
                                        break
                    except Exception as e:
                        logger.warning(f"Strategy 2 failed: {str(e)}")
                
                break
        
        # Fallback if parsing fails
        if therapist_speaker is None or client_speaker is None:
            # Simple frequency analysis
            logger.warning("Using fallback speaker assignment method")
            speaker_counts = {}
            for utterance in transcript["utterances"]:
                speaker = utterance["speaker"]
                if speaker not in speaker_counts:
                    speaker_counts[speaker] = 0
                speaker_counts[speaker] += 1
            
            # Sort speakers by frequency (most frequent is likely the therapist)
            speakers_sorted = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)
            if len(speakers_sorted) >= 2:
                therapist_speaker = speakers_sorted[0][0]
                client_speaker = speakers_sorted[1][0]
            elif len(speakers_sorted) == 1:
                therapist_speaker = speakers_sorted[0][0]
                client_speaker = None
        
        logger.info(f"Speaker assignments: Therapist={therapist_speaker}, Client={client_speaker}")
        
        # Update speaker labels in the transcript
        for utterance in transcript["utterances"]:
            if therapist_speaker and utterance["speaker"] == therapist_speaker:
                utterance["speaker"] = therapist_name
            elif client_speaker and utterance["speaker"] == client_speaker:
                utterance["speaker"] = patient_name
            else:
                # For any unassigned speakers
                if "speaker" in utterance and utterance["speaker"].isalpha() and len(utterance["speaker"]) == 1:
                    # If it's just a single letter (typical diarization output)
                    if utterance["speaker"] not in [therapist_speaker, client_speaker]:
                        # Default to patient for unknown speakers
                        utterance["speaker"] = patient_name
        
        # Create output directory
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get Audio folder path
        output_dir = os.path.join(current_dir, "speaker_assigned_transcript")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename
        input_path = Path(transcript_path)
        output_filename = f"{input_path.stem}_assigned{input_path.suffix}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the updated transcript
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        
        print(f"Success! Transcript with assigned roles saved to: {os.path.abspath(output_path)}")
        return output_path
        
    except Exception as e:
        error_msg = f"Error assigning speaker roles: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e


def main():
    assign_speaker_roles("transcriptions/Mock Therapy Convo with Dimple_transcript.json", "Nate", "Dimple")


if __name__ == "__main__":
    main()

