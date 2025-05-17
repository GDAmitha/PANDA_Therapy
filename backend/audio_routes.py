"""
Audio processing routes for PANDA Therapy API

This module handles audio file uploads, processing, and integration with the RAG system.
"""

from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
import json
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import models and authentication
from backend.models.user import User
from backend.simple_auth import get_current_user
from backend.database import Database

# Import audio processing components
try:
    from backend.Audio.audio_compat import is_audio_processing_available, process_audio_file
    audio_processing_available = is_audio_processing_available()
except ImportError:
    logging.warning("Audio processing components not available")
    audio_processing_available = False

# Import RAG components
try:
    from backend.rag_agent import TherapyRAGAgent
    rag_available = True
    rag_agent = TherapyRAGAgent()
except ImportError:
    rag_available = False
    rag_agent = None

# Create router
router = APIRouter(tags=["audio"])

# Initialize database
db = Database()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to format transcript for RAG
def format_transcript_for_rag(transcript, session_id):
    """Format transcript data for RAG ingestion"""
    transcript_text = ""
    
    for i, entry in enumerate(transcript):
        speaker = entry.get("speaker", "Unknown")
        text = entry.get("text", "")
        emotion = entry.get("emotion", "neutral")
        
        # Add formatted entry to transcript text
        transcript_text += f"{speaker} [{emotion}]: {text}\n"
        
    # Add session metadata
    transcript_text = f"Session ID: {session_id}\n\n" + transcript_text
    
    return transcript_text

@router.post("/audio/upload")
async def upload_audio(
    file: UploadFile = File(...),
    therapist_name: str = Form("Therapist"),
    patient_name: str = Form("Patient"),
    current_user: User = Depends(get_current_user)
):
    """
    Process uploaded audio file, perform emotion analysis, and store in database
    """
    try:
        # Generate a session ID
        session_id = str(uuid.uuid4())
        
        # Create output directory for processed files if it doesn't exist
        audio_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed_audio")
        os.makedirs(audio_output_dir, exist_ok=True)
        
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            contents = await file.read()
            with open(temp_file.name, 'wb') as f:
                f.write(contents)
            
            # Process the audio file if audio processing is available
            if audio_processing_available:
                try:
                    # Process the audio file and get transcript with emotion analysis
                    output_path = os.path.join(audio_output_dir, f"{session_id}.json")
                    processing_result = process_audio_file(
                        temp_file.name, 
                        output_path, 
                        therapist_name=therapist_name, 
                        patient_name=patient_name
                    )
                    
                    # Load the processed result
                    with open(output_path, 'r') as f:
                        transcript_data = json.load(f)
                    
                    # Store the session in the database
                    session_data = {
                        "session_id": session_id,
                        "therapist": therapist_name,
                        "patient": patient_name,
                        "created_at": datetime.now().isoformat(),
                        "audio_path": output_path,
                    }
                    
                    db_session_id = db.create_therapy_session(current_user.id, session_data)
                    
                    # Store the transcript in the database
                    transcript_data["session_id"] = db_session_id
                    transcript_id = db.create_transcript(current_user.id, transcript_data)
                    
                    # Also add to the RAG system if available
                    if rag_available and rag_agent:
                        try:
                            # Format for RAG indexing
                            rag_document = format_transcript_for_rag(transcript_data.get("transcript", []), session_id)
                            
                            # Add document to the agent
                            from llama_index.core import Document
                            doc = Document(
                                text=rag_document,
                                metadata={
                                    "source": "audio_transcript",
                                    "session_id": session_id,
                                    "user_id": current_user.id
                                }
                            )
                            
                            # Add the document to the RAG agent
                            rag_agent.index.insert(doc)
                            logger.info(f"Added processed audio transcript to RAG index for user {current_user.id}")
                        except Exception as e:
                            logger.error(f"Failed to add transcript to RAG index: {str(e)}")
                    
                    return {
                        "status": "success",
                        "message": "Audio file processed with emotion analysis",
                        "session_id": session_id,
                        "transcript_id": transcript_id,
                        "transcript_count": len(transcript_data.get("transcript", []))
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing audio file: {str(e)}")
                    # Fall back to simulated processing on error
            
            # If audio processing is not available or failed, create a simulated transcript
            logger.warning("Using simulated transcript - audio processing not available")
            simulated_transcript = [
                {"speaker": therapist_name, "text": "Hello, how are you feeling today?", "emotion": "neutral"},
                {"speaker": patient_name, "text": "I'm feeling a bit anxious about work.", "emotion": "anxious"},
                {"speaker": therapist_name, "text": "Can you tell me more about what's causing your anxiety?", "emotion": "concerned"},
                {"speaker": patient_name, "text": "I have a big presentation coming up and I'm worried I'll mess it up.", "emotion": "worried"},
                {"speaker": therapist_name, "text": "That's understandable. Let's talk about some strategies to help with presentation anxiety.", "emotion": "supportive"},
            ]
            
            # Store the simulated session in the database
            session_data = {
                "session_id": session_id,
                "therapist": therapist_name,
                "patient": patient_name,
                "created_at": datetime.now().isoformat(),
                "simulated": True,
            }
            
            db_session_id = db.create_therapy_session(current_user.id, session_data)
            
            # Store the simulated transcript in the database
            transcript_data = {
                "session_id": db_session_id,
                "transcript": simulated_transcript,
                "simulated": True,
            }
            
            transcript_id = db.create_transcript(current_user.id, transcript_data)
            
            # Also add to the RAG system if available
            if rag_available and rag_agent:
                try:
                    # Format for RAG indexing
                    rag_document = format_transcript_for_rag(simulated_transcript, session_id)
                    
                    # Add document to the agent
                    from llama_index.core import Document
                    doc = Document(
                        text=rag_document,
                        metadata={
                            "source": "simulated_audio_transcript",
                            "session_id": session_id,
                            "user_id": current_user.id
                        }
                    )
                    
                    # Add the document to the RAG agent
                    rag_agent.index.insert(doc)
                    logger.info(f"Added simulated transcript to RAG index for user {current_user.id}")
                except Exception as e:
                    logger.error(f"Failed to add transcript to RAG index: {str(e)}")
            
            return {
                "status": "success",
                "message": "Audio file received and simulated transcript generated",
                "session_id": session_id,
                "transcript_id": transcript_id,
                "transcript": simulated_transcript
            }
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")
    
    except Exception as e:
        logger.exception(f"Error uploading audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process audio file: {str(e)}"
        ) 