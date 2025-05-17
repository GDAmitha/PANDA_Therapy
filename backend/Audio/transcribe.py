import os
import assemblyai as aai
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple

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
            print("Running full ASR + diarization pipeline…")
            
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

# Example usage when run directly
if __name__ == "__main__":
    # Default audio file path for testing
    test_audio_file = "/Users/natedamstra/Downloads/Mock Therapy Convo with Dimple.wav"
    transcribe_audio(test_audio_file)
