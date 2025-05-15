"""
Audio Processing Module for Therapy Sessions

This module manages the processing of audio therapy sessions, including:
- Transcription
- Speaker diarization
- Emotion analysis
- Integration with the RAG system
"""

import os
import json
import logging
import tempfile
import shutil
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Processes therapy audio sessions and prepares them for the RAG system."""
    
    def __init__(self, storage_dir: str = "./audio_data"):
        """
        Initialize the audio processor.
        
        Args:
            storage_dir: Directory to store processed audio data
        """
        self.storage_dir = storage_dir
        self.transcription_dir = os.path.join(storage_dir, "transcriptions")
        self.speaker_assign_dir = os.path.join(storage_dir, "speaker_assignments")
        self.emotion_analysis_dir = os.path.join(storage_dir, "emotion_analysis")
        
        # Create directories if they don't exist
        for directory in [self.storage_dir, self.transcription_dir, 
                          self.speaker_assign_dir, self.emotion_analysis_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def process_audio_file(self, audio_file_path: str, 
                          therapist_name: str = "Therapist", 
                          patient_name: str = "Patient") -> Dict[str, Any]:
        """
        Process an audio therapy session through the entire pipeline.
        
        Args:
            audio_file_path: Path to the audio file
            therapist_name: Name of the therapist in the session
            patient_name: Name of the patient in the session
            
        Returns:
            Dict with session info and processing results
        """
        try:
            # Use the updated audio module
            from .Audio.modules.pipeline import TherapyAudioPipeline
            
            # Create a pipeline instance
            pipeline = TherapyAudioPipeline(base_data_dir=self.storage_dir)
            
            # Process the audio through the entire pipeline
            logger.info(f"Processing audio file through pipeline: {audio_file_path}")
            result = pipeline.process_audio(
                audio_file_path,
                therapist_name=therapist_name,
                patient_name=patient_name
            )
            
            if not result.get("success", False):
                error = result.get("error", "Unknown processing error")
                logger.error(f"Audio processing failed: {error}")
                return {
                    "success": False,
                    "error": error
                }
            
            # Extract paths and data from the result
            files = result.get("files", {})
            emotion_data = result.get("emotion_data", {})
            
            return {
                "success": True,
                "session_id": result.get("session_id", ""),
                "audio_file": audio_file_path,
                "transcript_path": files.get("transcript", ""),
                "speaker_assignment_path": files.get("speaker_assignment", ""),
                "emotion_analysis_path": files.get("emotion_analysis", ""),
                "data": emotion_data
            }
            
        except Exception as e:
            logger.exception(f"Error processing audio file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def prepare_for_rag(self, emotion_analysis_path: str) -> List[Dict[str, Any]]:
        """
        Convert emotion analysis results to chunks suitable for RAG ingestion.
        
        Args:
            emotion_analysis_path: Path to the emotion analysis JSON file
            
        Returns:
            List of chunks ready for RAG indexing
        """
        try:
            # Use the pipeline module's helper function
            from .Audio.modules.pipeline import TherapyAudioPipeline
            
            # Create a pipeline instance
            pipeline = TherapyAudioPipeline()
            
            # Load emotion analysis data
            with open(emotion_analysis_path, 'r') as f:
                emotion_data = json.load(f)
            
            # Use the pipeline to prepare RAG chunks
            rag_chunks = pipeline.prepare_rag_chunks(emotion_data)
            
            return rag_chunks
            
        except Exception as e:
            logger.exception(f"Error preparing RAG chunks: {str(e)}")
            return []
            
    def process_audio_session(self, file_path: str, therapist_name: str = "Therapist", 
                             patient_name: str = "Patient") -> Dict[str, Any]:
        """
        Process an entire audio session and prepare it for RAG.
        
        Args:
            file_path: Path to the audio file
            therapist_name: Name of the therapist
            patient_name: Name of the patient
            
        Returns:
            Dict with processing results and RAG-ready chunks
        """
        # Use the updated audio pipeline
        from .Audio.modules.pipeline import TherapyAudioPipeline
        
        try:
            # Create pipeline instance
            pipeline = TherapyAudioPipeline(base_data_dir=self.storage_dir)
            
            # Process the audio
            logger.info(f"Processing audio session: {file_path}")
            result = pipeline.process_audio(
                file_path,
                therapist_name=therapist_name,
                patient_name=patient_name
            )
            
            if not result.get("success", False):
                return {"success": False, "error": result.get("error", "Unknown error")}
            
            # Get the emotion data and prepare RAG chunks
            emotion_data = result.get("emotion_data", {})
            chunks = pipeline.prepare_rag_chunks(emotion_data)
            
            return {
                "success": True,
                "session_id": result.get("session_id", ""),
                "chunks": chunks,
                "file_paths": result.get("files", {})
            }
            
        except Exception as e:
            logger.exception(f"Error in audio session processing: {str(e)}")
            return {"success": False, "error": str(e)}
