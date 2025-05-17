"""
Speaker role assignment module for therapy session transcripts.
Assigns meaningful speaker names to the transcript utterances.
"""
import os
import json
from typing import Dict, Optional, List, Any

def assign_speaker_roles(
    transcript_data: Dict[str, Any],
    therapist_name: str = "Therapist",
    patient_name: str = "Patient",
    output_dir: str = "../data/speaker_assignments"
) -> Dict[str, Any]:
    """
    Assign meaningful names to speakers in the transcript and save to a JSON file.
    
    Args:
        transcript_data: Raw transcript data with speaker labels (A, B, etc.)
        therapist_name: Name to assign to the therapist (typically first speaker)
        patient_name: Name to assign to the patient
        output_dir: Directory to save the speaker-assigned transcript
    
    Returns:
        Dict with the speaker-assigned transcript data
    """
    # Early return if there are no utterances
    if "utterances" not in transcript_data or not transcript_data["utterances"]:
        return transcript_data
        
    # Extract unique speakers
    unique_speakers = set()
    for utterance in transcript_data["utterances"]:
        if "speaker" in utterance:
            unique_speakers.add(utterance["speaker"])
    
    # Create a mapping from speaker label to assigned name
    # Assuming first speaker is therapist in a therapy context
    speaker_map = {}
    speaker_list = sorted(list(unique_speakers))
    
    if len(speaker_list) >= 1:
        speaker_map[speaker_list[0]] = therapist_name
    
    if len(speaker_list) >= 2:
        speaker_map[speaker_list[1]] = patient_name
    
    # For any additional speakers, assign generic names
    for i, speaker in enumerate(speaker_list[2:], start=1):
        speaker_map[speaker] = f"Other_{i}"
    
    # Assign the new names to utterances
    assigned_transcript = {"utterances": []}
    
    for utterance in transcript_data["utterances"]:
        speaker_label = utterance.get("speaker", "Unknown")
        assigned_name = speaker_map.get(speaker_label, f"Speaker_{speaker_label}")
        
        # Create a new utterance with the assigned name
        new_utterance = utterance.copy()
        new_utterance["speaker"] = assigned_name
        assigned_transcript["utterances"].append(new_utterance)
    
    # Copy the full text if available
    if "text" in transcript_data:
        assigned_transcript["text"] = transcript_data["text"]
    
    # Save the assigned transcript to a file if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine the output filename
        if isinstance(transcript_data, dict) and "filename" in transcript_data:
            base_name = os.path.splitext(transcript_data["filename"])[0]
        else:
            import datetime
            base_name = f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = os.path.join(output_dir, f"{base_name}_assigned.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(assigned_transcript, f, indent=2, ensure_ascii=False)
            
        print(f"Speaker-assigned transcript saved to: {output_path}")
        assigned_transcript["filename"] = output_path
    
    return assigned_transcript
    
# Example usage
if __name__ == "__main__":
    # Test with a sample transcript
    sample_transcript = {
        "text": "Speaker A: Hello, how are you today? Speaker B: I'm feeling anxious.",
        "utterances": [
            {"speaker": "A", "text": "Hello, how are you today?", "start": 0, "end": 2000},
            {"speaker": "B", "text": "I'm feeling anxious.", "start": 2500, "end": 4000}
        ]
    }
    
    assigned = assign_speaker_roles(
        sample_transcript,
        therapist_name="Dr. Smith",
        patient_name="Jane"
    )
    
    print("\nAssigned transcript:")
    for utterance in assigned["utterances"]:
        print(f"{utterance['speaker']}: {utterance['text']}")
