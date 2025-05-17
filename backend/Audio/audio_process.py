"""
Audio processing module for PANDA Therapy

This module orchestrates the audio processing pipeline:
1. Transcription
2. Speaker role assignment
3. Emotion analysis from audio
4. Emotion analysis from text
"""

import os
import sys
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import necessary functions directly
from .transcribe_to_json import transcribe_audio_to_json
from .assign_speaker_roles import assign_speaker_roles
from .assign_audio_emotions import assign_audio_emotions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_audio_file(
    input_path: str, 
    output_path: str, 
    therapist_name: str = "Therapist", 
    patient_name: str = "Patient"
) -> Dict[str, Any]:
    """
    Process an audio file through the entire pipeline:
    1. Transcription
    2. Speaker role assignment
    3. Emotion analysis
    
    Args:
        input_path: Path to the input audio file
        output_path: Path to save the processed result
        therapist_name: Name of the therapist
        patient_name: Name of the patient
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing audio file: {input_path}")
    
    # Create directories for each processing stage in the Audio folder
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get Audio folder path
    transcriptions_dir = os.path.join(current_dir, "transcriptions")
    speakers_dir = os.path.join(current_dir, "speaker_assigned_transcript")
    emotions_dir = os.path.join(current_dir, "audio_emo_transcript")
    
    # Create all directories if they don't exist
    os.makedirs(transcriptions_dir, exist_ok=True)
    os.makedirs(speakers_dir, exist_ok=True)
    os.makedirs(emotions_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create paths for intermediate outputs
    file_name = os.path.basename(input_path).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = os.path.join(transcriptions_dir, f"{file_name}_transcript_{timestamp}.json")
    speaker_path = os.path.join(speakers_dir, f"{file_name}_speakers_{timestamp}.json")
    emotion_path = os.path.join(emotions_dir, f"{file_name}_emotions_{timestamp}.json")
    
    try:
        # Step 1: Transcribe the audio file
        logger.info("Step 1: Transcribing audio...")
        success, transcript_result = transcribe_audio_to_json(input_path)
        
        if not success:
            logger.error(f"Transcription failed: {transcript_result}")
            return {"error": f"Transcription failed: {transcript_result}"}
        
        # The transcript_result should be the path to the transcript JSON file
        logger.info(f"Transcription completed: {transcript_result}")
        
        # Step 2: Assign speaker roles
        logger.info("Step 2: Assigning speaker roles...")
        try:
            logger.info("Assigning speaker roles to transcript")
            logger.info(f"Using therapist name: {therapist_name}, patient name: {patient_name}")
            speaker_result = assign_speaker_roles(transcript_result, therapist_name, patient_name)
            logger.info(f"Speaker roles assigned successfully, saved to: {speaker_result}")
        except Exception as e:
            logger.error(f"Error assigning speaker roles: {str(e)}")
            speaker_result = transcript_result  # Use the original transcript if speaker assignment fails

        # Step 3: Assign Audio Emotions
        try:
            logger.info("Analyzing audio emotions")
            # Make sure to use the same audio file that was used for transcription
            emotion_result = assign_audio_emotions(input_path, speaker_result)
            logger.info(f"Audio emotions analyzed successfully, saved to: {emotion_result}")
            
            # Verify the emotion file exists
            if not os.path.exists(emotion_result):
                logger.error(f"Emotion result file does not exist: {emotion_result}")
                emotion_result = speaker_result  # Use the speaker transcript if emotion analysis fails
        except Exception as e:
            logger.error(f"Error analyzing audio emotions: {str(e)}")
            emotion_result = speaker_result  # Use the speaker transcript if emotion analysis fails
            
        # Load the final result with emotion analysis
        try:
            with open(emotion_result, 'r') as f:
                final_result = json.load(f)
                
            # Enhance the result with structured metadata
            enhanced_result = _add_structured_metadata(
                transcript_data=final_result,
                audio_file_path=input_path,
                therapist_name=therapist_name,
                client_name=patient_name
            )
                
            # Copy to the desired output path
            with open(output_path, 'w') as f:
                json.dump(enhanced_result, f, indent=2)
                
            logger.info(f"Audio processing completed successfully. Final output saved to {output_path}")
            return enhanced_result
        except Exception as e:
            logger.error(f"Error saving final results: {str(e)}")
            return {"error": f"Error saving final results: {str(e)}", "emotion_result": emotion_result}
    
    except ImportError as e:
        logger.warning(f"Import error during processing: {str(e)}")
        logger.warning("Trying fallback with AssemblyAI API...")
        return process_audio_file_fallback(input_path, output_path, therapist_name, patient_name)
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        # Try the fallback method
        logger.warning("Trying fallback with AssemblyAI API...")
        try:
            return process_audio_file_fallback(input_path, output_path, therapist_name, patient_name)
        except Exception as fallback_error:
            logger.error(f"Fallback processing also failed: {str(fallback_error)}")
            # Create an output file with error information
            error_result = {
                "error": f"Processing failed: {str(e)}. Fallback also failed: {str(fallback_error)}",
                "transcript": [],
                "emotion_analysis": []
            }
            try:
                with open(output_path, 'w') as f:
                    json.dump(error_result, f, indent=2)
            except Exception as write_error:
                logger.error(f"Error writing output file: {str(write_error)}")
            
            return error_result

def process_audio_file_fallback(
    input_path: str, 
    output_path: str, 
    therapist_name: str = "Therapist", 
    patient_name: str = "Patient"
) -> Dict[str, Any]:
    """
    Fallback method using AssemblyAI API for processing when local components aren't available
    """
    try:
        import assemblyai as aai
        from datetime import datetime
        
        # Get the API key
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
        
        # Initialize AssemblyAI client
        aai.settings.api_key = api_key
        
        logger.info("Using AssemblyAI for audio processing")
        
        # Create a transcriber
        transcriber = aai.Transcriber()
        
        # Configure transcription options
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            auto_highlights=True,
            entity_detection=True,
            sentiment_analysis=True,
            iab_categories=True
        )
        
        # Start transcription
        logger.info(f"Uploading file to AssemblyAI: {input_path}")
        transcript = transcriber.transcribe(
            input_path,
            config=config
        )
        
        # Check if transcription was successful
        if transcript.status != "completed":
            raise RuntimeError(f"Transcription failed with status: {transcript.status}")
        
        # Format the transcript
        formatted_transcript = []
        for utterance in transcript.utterances:
            formatted_transcript.append({
                "speaker": f"{therapist_name if utterance.speaker == 'A' else patient_name}",
                "text": utterance.text,
                "start": utterance.start,
                "end": utterance.end
            })
        
        # Extract emotions based on sentiment analysis
        emotion_analysis = []
        for i, utterance in enumerate(transcript.utterances):
            # Map Assembly AI sentiment to emotions
            sentiment = "neutral"
            if hasattr(transcript, "sentiment_analysis_results"):
                # Find sentiment results that overlap with this utterance
                for result in transcript.sentiment_analysis_results:
                    if (result.start >= utterance.start and result.start <= utterance.end) or \
                       (result.end >= utterance.start and result.end <= utterance.end):
                        sentiment = result.sentiment
                        break
            
            # Map sentiment to emotion
            emotion_map = {
                "positive": "happiness",
                "negative": "sadness",
                "neutral": "neutral"
            }
            emotion = emotion_map.get(sentiment, "neutral")
            
            # Add to emotion analysis
            emotion_analysis.append({
                "speaker": f"{therapist_name if utterance.speaker == 'A' else patient_name}",
                "text": utterance.text,
                "predicted_wav_emotion": emotion,
                "predicted_text_emotion": emotion,
                "confidence": 0.7,
                "text_confidence": 0.7
            })
        
        # Combine results
        result = {
            "transcript": formatted_transcript,
            "emotion_analysis": emotion_analysis,
            "processed_at": datetime.now().isoformat(),
            "processing_method": "assemblyai"
        }
        
        # Enhance the result with structured metadata
        enhanced_result = _add_structured_metadata(
            transcript_data=result,
            audio_file_path=input_path,
            therapist_name=therapist_name,
            client_name=patient_name
        )
        
        # Save the enhanced result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(enhanced_result, f, indent=2)
        
        logger.info(f"AssemblyAI processing completed successfully. Output saved to {output_path}")
        return enhanced_result
        
    except ImportError:
        logger.warning("AssemblyAI not available, using simulated results")
        return _generate_simulated_results(input_path, output_path, therapist_name, patient_name)
    except Exception as e:
        logger.error(f"Error with AssemblyAI processing: {str(e)}")
        return _generate_simulated_results(input_path, output_path, therapist_name, patient_name)

def _generate_simulated_results(
    input_path: str, 
    output_path: str, 
    therapist_name: str, 
    patient_name: str
) -> Dict[str, Any]:
    """Generate simulated transcript and emotion analysis when no processing is available"""
    logger.warning(f"Generating simulated results for {input_path}")
    
    # Create a simulated transcript
    transcript = [
        {"speaker": therapist_name, "text": "Hello, how are you feeling today?", "start": 0.0, "end": 3.5},
        {"speaker": patient_name, "text": "I'm feeling a bit anxious about work.", "start": 4.0, "end": 8.5},
        {"speaker": therapist_name, "text": "Can you tell me more about what's causing your anxiety?", "start": 9.0, "end": 13.5},
        {"speaker": patient_name, "text": "I have a big presentation coming up and I'm worried I'll mess it up.", "start": 14.0, "end": 19.5},
        {"speaker": therapist_name, "text": "That's understandable. Let's talk about some strategies to help with presentation anxiety.", "start": 20.0, "end": 27.5},
        {"speaker": patient_name, "text": "I'd really appreciate that. I've been having trouble sleeping because of it.", "start": 28.0, "end": 33.5},
        {"speaker": therapist_name, "text": "Sleep difficulties are common with anxiety. Have you tried any relaxation techniques?", "start": 34.0, "end": 40.5},
        {"speaker": patient_name, "text": "I've tried deep breathing, but it's not helping much.", "start": 41.0, "end": 45.5},
        {"speaker": therapist_name, "text": "Let's explore some additional techniques that might work better for you.", "start": 46.0, "end": 52.5}
    ]
    
    # Create simulated emotion analysis
    emotion_analysis = [
        {"speaker": therapist_name, "text": "Hello, how are you feeling today?", "predicted_wav_emotion": "neutral", "predicted_text_emotion": "neutral", "confidence": 0.85, "text_confidence": 0.92},
        {"speaker": patient_name, "text": "I'm feeling a bit anxious about work.", "predicted_wav_emotion": "anxiety", "predicted_text_emotion": "anxiety", "confidence": 0.78, "text_confidence": 0.82},
        {"speaker": therapist_name, "text": "Can you tell me more about what's causing your anxiety?", "predicted_wav_emotion": "concern", "predicted_text_emotion": "concern", "confidence": 0.67, "text_confidence": 0.75},
        {"speaker": patient_name, "text": "I have a big presentation coming up and I'm worried I'll mess it up.", "predicted_wav_emotion": "fear", "predicted_text_emotion": "anxiety", "confidence": 0.88, "text_confidence": 0.91},
        {"speaker": therapist_name, "text": "That's understandable. Let's talk about some strategies to help with presentation anxiety.", "predicted_wav_emotion": "supportive", "predicted_text_emotion": "supportive", "confidence": 0.72, "text_confidence": 0.79},
        {"speaker": patient_name, "text": "I'd really appreciate that. I've been having trouble sleeping because of it.", "predicted_wav_emotion": "fatigue", "predicted_text_emotion": "worry", "confidence": 0.81, "text_confidence": 0.76},
        {"speaker": therapist_name, "text": "Sleep difficulties are common with anxiety. Have you tried any relaxation techniques?", "predicted_wav_emotion": "compassion", "predicted_text_emotion": "inquiry", "confidence": 0.69, "text_confidence": 0.74},
        {"speaker": patient_name, "text": "I've tried deep breathing, but it's not helping much.", "predicted_wav_emotion": "frustration", "predicted_text_emotion": "disappointment", "confidence": 0.73, "text_confidence": 0.84},
        {"speaker": therapist_name, "text": "Let's explore some additional techniques that might work better for you.", "predicted_wav_emotion": "encouraging", "predicted_text_emotion": "hopeful", "confidence": 0.77, "text_confidence": 0.86}
    ]
    
    # Create the basic result
    result = {
        "transcript": transcript,
        "emotion_analysis": emotion_analysis,
        "simulated": True
    }
    
    # Create directories for each processing stage in the Audio folder
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get Audio folder path
    transcriptions_dir = os.path.join(current_dir, "transcriptions")
    speakers_dir = os.path.join(current_dir, "speaker_assigned_transcript")
    emotions_dir = os.path.join(current_dir, "audio_emo_transcript")
    
    # Create all directories if they don't exist
    os.makedirs(transcriptions_dir, exist_ok=True)
    os.makedirs(speakers_dir, exist_ok=True)
    os.makedirs(emotions_dir, exist_ok=True)
    
    # Enhance with structured metadata
    enhanced_result = _add_structured_metadata(
        transcript_data=result,
        audio_file_path=input_path,
        therapist_name=therapist_name,
        client_name=patient_name
    )
    
    # Also mark as simulated in the structured format
    enhanced_result["simulated"] = True
    if "session_info" in enhanced_result:
        enhanced_result["session_info"]["simulated"] = True
    
    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(enhanced_result, f, indent=2)
    
    logger.info(f"Simulated results saved to {output_path}")
    return enhanced_result

def _add_structured_metadata(
    transcript_data: Dict[str, Any],
    audio_file_path: str,
    therapist_name: str,
    client_name: str
) -> Dict[str, Any]:
    """
    Enhance transcript data with structured metadata format.
    
    Args:
        transcript_data: The original transcript data dictionary
        audio_file_path: Path to the original audio file
        therapist_name: Name of the therapist
        client_name: Name of the client
        
    Returns:
        Enhanced transcript data with structured metadata
    """
    # Generate a unique session ID if not present
    session_id = transcript_data.get('session_id', str(int(datetime.now().timestamp())))
    
    # Create structured transcript
    structured_data = {
        "session_id": session_id,
        "therapist": {
            "id": f"therapist_{session_id}",
            "name": therapist_name
        },
        "client": {
            "id": f"client_{session_id}",
            "name": client_name
        },
        "session_info": {
            "date": datetime.now().isoformat(),
            "audio_file": os.path.basename(audio_file_path),
            "processed_at": datetime.now().timestamp(),
            "duration_minutes": _estimate_session_duration(transcript_data),
            "topic_summary": _generate_topic_summary(transcript_data, client_name)
        },
        # Keep the original transcript and emotion analysis
        "transcript": transcript_data.get("transcript", []) or transcript_data.get("utterances", []),
        "emotion_analysis": transcript_data.get("emotion_analysis", [])
    }
    
    # Add processed_at timestamp (backward compatibility)
    structured_data["processed_at"] = structured_data["session_info"]["processed_at"]
    
    return structured_data


def _estimate_session_duration(transcript_data: Dict[str, Any]) -> int:
    """
    Estimate the session duration in minutes based on audio timestamps.
    
    Args:
        transcript_data: Transcript data dictionary
        
    Returns:
        Estimated duration in minutes
    """
    try:
        # Get transcript entries
        transcript = transcript_data.get("transcript", []) or transcript_data.get("utterances", [])
        
        if not transcript:
            return 30  # Default duration
            
        # Check if timestamps are available
        if "start" in transcript[0] and "end" in transcript[-1]:
            duration_ms = transcript[-1]["end"] - transcript[0]["start"]
            duration_min = max(1, round(duration_ms / 60000))  # Convert ms to min, minimum 1 minute
            return duration_min
    except Exception as e:
        logger.warning(f"Could not estimate session duration: {e}")
    
    # Default duration if estimation fails
    return 30  # Default therapy session length in minutes


def _generate_topic_summary(transcript_data: Dict[str, Any], client_name: str) -> str:
    """
    Generate a simple topic summary from the transcript.
    
    Args:
        transcript_data: Transcript data dictionary
        client_name: Name of the client to identify client utterances
        
    Returns:
        Topic summary as a string
    """
    try:
        # Get transcript entries
        transcript = transcript_data.get("transcript", []) or transcript_data.get("utterances", [])
        
        if not transcript:
            return "Therapy session"
        
        # Find the first substantial client response (likely describing their issue)
        for entry in transcript:
            if entry.get("speaker") == client_name and len(entry.get("text", "")) > 30:
                # Extract the first part of their response as topic
                text = entry.get("text", "")
                summary = text[:100] + "..." if len(text) > 100 else text
                return summary
    except Exception as e:
        logger.warning(f"Could not generate topic summary: {e}")
    
    return "Therapy session"


# If called directly, run a test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[1] + "_processed.json" if len(sys.argv) <= 2 else sys.argv[2]
        
        result = process_audio_file(input_file, output_file)
        print(f"Processed file saved to {output_file}")
    else:
        print("Usage: python audio_process.py <input_audio_file> [output_json_file]")
