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
            
    def process_and_vectorize(self, audio_file_path: str, client_name: str, therapist_name: str = "Therapist"):
        """
        Process an audio file and automatically vectorize it for the client
        
        Args:
            audio_file_path: Path to the audio file
            client_name: Name of the client (used for transcript retrieval)
            therapist_name: Name of the therapist (default: "Therapist")
            
        Returns:
            Success status
        """
        # First process the audio file
        output_path = self.process_audio(audio_file_path, client_name, therapist_name)
        if not output_path:
            return False
            
        # Then add to database and vectorize
        return self.add_to_database_and_vectorize(output_path, client_name)
        
    def add_to_database_and_vectorize(self, transcript_path: str, client_name: str) -> bool:
        """
        Add a processed transcript to the database and vectorize it for the client
        
        Args:
            transcript_path: Path to the processed transcript JSON
            client_name: Name of the client
            
        Returns:
            Success status
        """
        try:
            from backend.transcript_memory import TherapyTranscriptMemory
            import json
            import os
            
            # Load the transcript
            with open(transcript_path, 'r') as f:
                transcript_data = json.load(f)
                
            # Get database path
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "transcripts.json")
            
            # Load existing database
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    try:
                        db = json.load(f)
                    except json.JSONDecodeError:
                        db = {}
            else:
                db = {}
                
            # Generate a new session ID
            session_id = str(max([int(k) for k in db.keys() if k.isdigit()] + [0]) + 1)
            
            # Add the transcript to the database
            db[session_id] = transcript_data
            
            # Save the updated database
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with open(db_path, 'w') as f:
                json.dump(db, f, indent=2)
                
            # Now vectorize the transcript for this client
            memory = TherapyTranscriptMemory(client_name)
            success = memory.create_memory_index(force_reload=True)
            
            if success:
                logger.info(f"Successfully added and vectorized transcript for client {client_name}")
                return True
            else:
                logger.error(f"Failed to vectorize transcript for client {client_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding transcript to database and vectorizing: {e}")
            return False
    
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
