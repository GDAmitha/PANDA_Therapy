# Main entry point for clients interaction with chatbot
from transcribe import transcribe_audio

import os
import json
from pathlib import Path
from letta_client import Letta

# Initialize the Letta client
client = Letta(token=os.getenv("LETTA_API_KEY"))
agent_assign_roles = client.agents.retrieve(agent_id="agent-3f300241-137e-43bc-bea8-e7ee3185e973")


def assign_speaker_roles(transcript_path: str, therapist_name: str = "Therapist", patient_name: str = "Client") -> str:
    """
    Assign speaker roles to a transcript and save the result to a new JSON file.
    
    Args:
        transcript_path (str): Path to the transcript JSON file
        therapist_name (str): Name to use for the therapist speaker
        patient_name (str): Name to use for the patient/client speaker
        
    Returns:
        str: Path to the output JSON file with assigned speaker roles
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        json.JSONDecodeError: If the input file is not valid JSON
    """
    # Validate input file
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    
    # Load the transcript
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON file: {transcript_path}", e.doc, e.pos)
    
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
        for message in response.messages:
            if message.message_type == "assistant_message":
                result = message.content.strip()
                parts = result.split(',')
                therapist_speaker = parts[0].split('=')[1].strip().upper()
                client_speaker = parts[1].split('=')[1].strip().upper()
                break
        
        # Update speaker labels in the transcript
        for utterance in transcript["utterances"]:
            if utterance["speaker"] == therapist_speaker:
                utterance["speaker"] = therapist_name
            elif utterance["speaker"] == client_speaker:
                utterance["speaker"] = patient_name
        
        # Create output directory if it doesn't exist
        output_dir = "speaker_assign_transcript"
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

