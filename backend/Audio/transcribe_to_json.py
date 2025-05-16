# Main entry point for clients interaction with chatbot
from transcribe import transcribe_audio

import os
import json
import argparse
import sys
from typing import Dict, Optional, Tuple
from transcribe import transcribe_audio

def transcribe_audio_to_json(
    audio_file_path: str, 
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    Transcribe an audio file and save the results to a JSON file in the 'transcriptions' folder.
    
    Args:
        audio_file_path (str): Path to the audio file to transcribe
        verbose (bool): Whether to print progress information
        
    Returns:
        Tuple[bool, str]: 
            - Boolean indicating success or failure
            - Path to the output JSON file if successful, error message if failed
    """
    # Validate input file
    if not os.path.isfile(audio_file_path):
        return False, f"Audio file not found: {audio_file_path}"
    
    # Create transcriptions directory in the Audio folder if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get Audio folder path
    output_dir = os.path.join(current_dir, "transcriptions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_transcript.json")
    
    if verbose:
        print(f"Starting transcription of: {audio_file_path}")
        print(f"Output will be saved to: {output_path}")
    
    # Transcribe the audio file
    try:
        result, error = transcribe_audio(audio_file_path, verbose=verbose)
        
        if error:
            return False, f"Transcription failed: {error}"
    
    except Exception as e:
        return False, f"Failed to transcribe audio: {str(e)}"
    
    try:
        # Import speaker role assignment if available
        try:
            from assign_speaker_roles import assign_speaker_roles
            if verbose:
                print("Assigning speaker roles...")
            result = assign_speaker_roles(result)
        except ImportError:
            if verbose:
                print("Speaker role assignment module not found. Using default speaker labels.")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not assign speaker roles: {str(e)}")
        
        # Save the result to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"\nTranscript successfully saved to: {output_path}")
        
        return True, output_path
        
    except Exception as e:
        error_msg = f"Failed to process transcript: {str(e)}"
        if verbose:
            print(error_msg)
        return False, error_msg


def main():
    """Command-line interface for transcribing audio files."""
    parser = argparse.ArgumentParser(description='Transcribe an audio file with speaker diarization.')
    parser.add_argument('audio_file', help='Path to the audio file to transcribe')
    parser.add_argument('-q', '--quiet', action='store_true', 
                      help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Handle file path with spaces
    audio_file_path = ' '.join(args.audio_file) if isinstance(args.audio_file, list) else args.audio_file
    
    success, result = transcribe_audio_to_json(
        audio_file_path=audio_file_path,
        verbose=not args.quiet
    )
    
    if not success:
        print(f"Error: {result}", file=sys.stderr)
        return 1
    
    print(f"Success! Transcript saved to: {os.path.abspath(result)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

# 1. Process Audio
# 1.1 Receive audio file from user
# 1.2 Transcribe audio file
# 1.3 Assign speaker labels
# 1.4 Add sentiment analysis
# 1.5 Add topic modeling
# 2. Send transcribed text to Letta
# 3. 
