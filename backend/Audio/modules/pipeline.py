"""
Complete audio processing pipeline for therapy sessions.
Coordinates transcription, speaker assignment, and emotion analysis.
"""
import os
import json
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapyAudioPipeline:
    """Manages the complete audio processing pipeline for therapy sessions."""
    
    def __init__(self, base_data_dir: str = "../data"):
        """
        Initialize the pipeline with paths for data.
        
        Args:
            base_data_dir: Base directory for all data outputs
        """
        self.base_data_dir = base_data_dir
        self.transcription_dir = os.path.join(base_data_dir, "transcriptions")
        self.speaker_dir = os.path.join(base_data_dir, "speaker_assignments")
        self.emotion_dir = os.path.join(base_data_dir, "emotion_analysis")
        
        # Create directories if they don't exist
        for directory in [self.transcription_dir, self.speaker_dir, self.emotion_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def process_audio(
        self, 
        audio_file_path: str,
        therapist_name: str = "Therapist",
        patient_name: str = "Patient",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Process an audio file through the complete pipeline.
        
        Args:
            audio_file_path: Path to the audio file
            therapist_name: Name to assign to the therapist
            patient_name: Name to assign to the patient
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with processing results and file paths
        """
        try:
            # Step 1: Transcribe audio
            from .transcriber import transcribe_audio_to_json
            if verbose:
                logger.info(f"Transcribing audio: {audio_file_path}")
            
            success, transcript_path = transcribe_audio_to_json(
                audio_file_path, 
                verbose=verbose,
                output_dir=self.transcription_dir
            )
            
            if not success:
                return {"success": False, "error": f"Transcription failed: {transcript_path}"}
                
            # Step 2: Assign speaker roles
            from .speaker_assignment import assign_speaker_roles
            if verbose:
                logger.info("Assigning speaker roles to transcript")
                
            with open(transcript_path, 'r') as f:
                transcript_data = json.load(f)
            
            speaker_result = assign_speaker_roles(
                transcript_data,
                therapist_name=therapist_name,
                patient_name=patient_name,
                output_dir=self.speaker_dir
            )
            
            speaker_path = speaker_result.get("filename")
            if not speaker_path:
                return {"success": False, "error": "Speaker assignment failed"}
            
            # Step 3: Analyze emotions
            from .emotion_analyzer import EmotionAnalyzer
            if verbose:
                logger.info("Analyzing emotions in audio and text")
                
            analyzer = EmotionAnalyzer()
            emotion_result = analyzer.analyze_emotions(
                audio_file_path,
                speaker_path,
                output_dir=self.emotion_dir
            )
            
            emotion_path = emotion_result.get("output_path")
            if not emotion_path:
                return {"success": False, "error": "Emotion analysis failed"}
            
            # Generate session ID
            session_id = f"session_{os.path.splitext(os.path.basename(audio_file_path))[0]}"
            
            # Return success with all paths
            return {
                "success": True,
                "session_id": session_id,
                "files": {
                    "audio": audio_file_path,
                    "transcript": transcript_path,
                    "speaker_assignment": speaker_path,
                    "emotion_analysis": emotion_path
                },
                "emotion_data": emotion_result
            }
            
        except Exception as e:
            logger.exception(f"Error in audio processing pipeline: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def prepare_rag_chunks(self, emotion_data: Dict[str, Any]) -> list:
        """
        Convert emotion analysis data to chunks suitable for RAG indexing.
        
        Args:
            emotion_data: Emotion analysis data dictionary
            
        Returns:
            List of chunks ready for RAG indexing
        """
        chunks = []
        
        # Extract session info
        audio_file = emotion_data.get("audio_file", "")
        session_id = f"session_{os.path.splitext(os.path.basename(audio_file))[0]}"
        
        # Create chunks from emotion analysis
        for i, item in enumerate(emotion_data.get("emotion_analysis", [])):
            text = item.get("text", "")
            speaker = item.get("speaker", "Unknown")
            wav_emotion = item.get("predicted_wav_emotion", "neutral")
            text_emotion = item.get("predicted_text_emotion", "neutral")
            
            # Create a summary with all the information
            summary = (
                f"{speaker} said: \"{text}\" "
                f"Voice emotion: {wav_emotion} "
                f"Text emotion: {text_emotion}"
            )
            
            # Create the chunk
            chunk = {
                "id": f"{session_id}_chunk_{i}",
                "summary": summary,
                "speaker": speaker,
                "emotion": wav_emotion,
                "text_emotion": text_emotion,
                "text": text,
                "timestamp": item.get("start", 0),
                "source": "audio_session",
                "audio_file": audio_file
            }
            
            chunks.append(chunk)
        
        return chunks

# For backward compatibility
def process_audio_session(
    audio_file_path: str,
    therapist_name: str = "Therapist",
    patient_name: str = "Patient"
) -> Dict[str, Any]:
    """
    Process an audio therapy session through the complete pipeline.
    
    Args:
        audio_file_path: Path to the audio file
        therapist_name: Name to assign to the therapist
        patient_name: Name to assign to the patient
        
    Returns:
        Dictionary with processing results including RAG-ready chunks
    """
    pipeline = TherapyAudioPipeline()
    result = pipeline.process_audio(
        audio_file_path,
        therapist_name=therapist_name,
        patient_name=patient_name
    )
    
    if not result.get("success", False):
        return result
    
    # Prepare RAG chunks
    emotion_data = result.get("emotion_data", {})
    chunks = pipeline.prepare_rag_chunks(emotion_data)
    
    result["chunks"] = chunks
    return result

# Example usage
if __name__ == "__main__":
    # Test the pipeline with a sample audio file
    audio_path = "/path/to/your/audio.wav"
    pipeline = TherapyAudioPipeline()
    result = pipeline.process_audio(audio_path, "Dr. Smith", "John")
