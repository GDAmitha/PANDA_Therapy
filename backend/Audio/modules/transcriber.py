"""
Audio transcription module that converts speech to text with speaker diarization
"""
import os
import json
from typing import Dict, Optional, Tuple, Any
import assemblyai as aai
from dotenv import load_dotenv

# Initialize the API key once when the module is imported
load_dotenv()
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not api_key:
    raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")

aai.settings.api_key = api_key

def transcribe_audio(audio_file_path: str, verbose: bool = True) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Transcribe an audio file using AssemblyAI's API with speaker diarization.
    
    Args:
        audio_file_path (str): Path to the audio file to transcribe
        verbose (bool): Whether to print progress information
        
    Returns:
        Tuple[Optional[Dict], Optional[str]]: 
            - A dictionary containing the full transcript and utterances if successful
            - An error message if an error occurred, None otherwise
    """
    if verbose:
        print(f"Starting transcription for file: {audio_file_path}")
        print("API Key found." if api_key.startswith("4aac28") else "Using API key from environment")

    # Verify file exists
    if not os.path.exists(audio_file_path):
        error_msg = f"Audio file not found at {audio_file_path}"
        if verbose:
            print(error_msg)
        return None, error_msg

    try:
        if verbose:
            print("Running full ASR + diarization pipeline...")
            
        config = aai.TranscriptionConfig(speaker_labels=True)
        transcriber = aai.Transcriber()
        
        if verbose:
            print("Starting transcription...")
            
        transcript = transcriber.transcribe(audio_file_path, config=config)
        
        if transcript.error:
            error_msg = f"Error during transcription: {transcript.error}"
            if verbose:
                print(error_msg)
            return None, error_msg
            
        # Prepare the result dictionary
        result = {
            "text": transcript.text,
            "utterances": [
                {"speaker": u.speaker, "text": u.text, "start": u.start, "end": u.end}
                for u in transcript.utterances
            ]
        }
        
        if verbose:
            print("\nTranscription completed successfully!")
            print("\nTranscript:")
            print("-" * 50)
            for utterance in result["utterances"]:
                print(f"Speaker {utterance['speaker']}: {utterance['text']}")
            print("-" * 50)
            print(f"\nFull transcript text:\n{result['text']}")
        
        return result, None
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        if verbose:
            print(error_msg)
        return None, error_msg

def save_transcript_to_json(transcript_data: Dict[str, Any], 
                           output_dir: str = "../data/transcriptions") -> str:
    """
    Save transcript data to a JSON file in the specified directory.
    
    Args:
        transcript_data: The transcript data to save
        output_dir: Directory where to save the transcript
        
    Returns:
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"transcript_{timestamp}.json")
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    print(f"Transcript saved to: {output_path}")
    return output_path

def transcribe_audio_to_json(
    audio_file_path: str, 
    verbose: bool = True,
    output_dir: str = "../data/transcriptions"
) -> Tuple[bool, str]:
    """
    Transcribe an audio file and save the results to a JSON file.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        verbose: Whether to print progress information
        output_dir: Directory where to save the transcript
        
    Returns:
        Tuple[bool, str]: 
            - Boolean indicating success or failure
            - Path to the output JSON file if successful, error message if failed
    """
    # Transcribe the audio
    result, error = transcribe_audio(audio_file_path, verbose=verbose)
    
    if error:
        return False, f"Transcription failed: {error}"
    
    try:
        # Save the result to JSON
        output_path = save_transcript_to_json(result, output_dir)
        return True, output_path
    
    except Exception as e:
        return False, f"Failed to save transcript: {str(e)}"

# Example usage when run directly
if __name__ == "__main__":
    # Default audio file path for testing
    test_audio_file = "/path/to/your/audio/file.wav"
    success, result = transcribe_audio_to_json(test_audio_file)
    print(f"Success: {success}, Result: {result}")
