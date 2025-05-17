"""
Chat and therapy session routes for PANDA Therapy
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import os
import logging
import tempfile
import shutil
import json

from models.user import User, Patient, Therapist, UserRole
from simple_auth import get_current_user
from combined_memory_connector import CombinedMemoryConnector
from transcript_memory import TherapyTranscriptMemory
from letta_agent import LettaAgentManager
from audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router and agent manager
router = APIRouter(tags=["chat"])
letta_mgr = LettaAgentManager()

# Dictionary to store user-specific memory connectors
user_memory_connectors = {}

# Audio processor
audio_processor = AudioProcessor()

def get_user_memory(user: User) -> CombinedMemoryConnector:
    """
    Get or create a combined memory connector for a specific user
    
    This connector provides two types of knowledge:
    1. General therapy knowledge (shared across all users)
    2. Patient-specific transcript memory (private to each user)
    
    Args:
        user: User to get memory connector for
        
    Returns:
        CombinedMemoryConnector instance for the user
    """
    if user.id not in user_memory_connectors:
        # Create a new combined memory connector for the user
        # This internally initializes both OpenAITherapyIndex and TherapyTranscriptMemory
        user_memory_connectors[user.id] = CombinedMemoryConnector(user.id)
        logger.info(f"Created new combined memory connector for user {user.id}")
    
    return user_memory_connectors[user.id]

@router.post("/chat")
async def chat(
    message: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Process a chat message using the user's RAG agent and/or Letta agent
    
    Args:
        message: User's chat message
        chat_history: Optional chat history
        current_user: Current authenticated user
        
    Returns:
        Response from the agent
    """
    try:
        # Get the user's combined memory connector
        memory_connector = get_user_memory(current_user)
        
        logger.info(f"Received chat request from user {current_user.id}: {message}")
        
        # Get combined knowledge from both general therapy documents and patient-specific transcripts
        therapy_knowledge = memory_connector.get_therapy_knowledge(message)
        transcript_memory = memory_connector.get_transcript_memory(message)
        
        # Combine the responses
        combined_response = ""
        if therapy_knowledge:
            combined_response += "THERAPY KNOWLEDGE:\n" + therapy_knowledge + "\n\n"
        if transcript_memory:
            combined_response += "PATIENT TRANSCRIPT MEMORY:\n" + transcript_memory
            
        if not combined_response:
            combined_response = "I don't have specific information about that. How can I help you?"
        
        # If user has a Letta agent, also process with that
        if current_user.letta_agent_id:
            try:
                # Format history for Letta agent
                formatted_history = []
                if chat_history:
                    for msg in chat_history:
                        if msg.get("role") == "user":
                            formatted_history.append({
                                "role": "human",
                                "content": msg["content"]
                            })
                        elif msg.get("role") == "assistant":
                            formatted_history.append({
                                "role": "ai",
                                "content": msg["content"]
                            })
                
                # Chat with Letta agent, providing combined context from both therapy documents and transcripts
                enhanced_message = f"[Context from knowledge system: {combined_response}]\n\nUser message: {message}"
                letta_result = letta_mgr.chat_with_agent(
                    current_user.letta_agent_id, 
                    enhanced_message, 
                    formatted_history
                )
                
                # Use Letta response if successful, otherwise fall back to combined memory response
                if "error" not in letta_result:
                    response = letta_result.get("response", combined_response)
                else:
                    response = combined_response
                    logger.warning(f"Letta agent error, using combined memory response: {letta_result.get('error')}")
            except Exception as e:
                # Fall back to combined memory response on error
                response = combined_response
                logger.error(f"Error with Letta agent, using combined memory response: {str(e)}")
        else:
            # No Letta agent, just use combined memory response
            response = combined_response
        
        return {"response": response, "user_id": current_user.id}
    
    except Exception as e:
        error_detail = f"Error in chat endpoint: {str(e)}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...), 
    therapist_name: str = Form("Therapist"), 
    patient_name: str = Form("Patient"),
    current_user: User = Depends(get_current_user)
):
    """
    Process an uploaded audio file and add to the user's knowledge base
    
    Args:
        file: Uploaded audio file
        therapist_name: Name of the therapist in the audio
        patient_name: Name of the patient in the audio
        current_user: Current authenticated user
        
    Returns:
        Processing results
    """
    try:
        logger.info(f"Processing audio file for user {current_user.id}: {file.filename}")
        
        # Save to the user's audio directory
        user_audio_dir = current_user.get_audio_path()
        os.makedirs(user_audio_dir, exist_ok=True)
        
        # Save to a temporary file first
        suffix = ".wav" if file.filename.endswith(".wav") else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Process the audio file
        result = audio_processor.process_audio_session(
            tmp_path,
            therapist_name,
            patient_name
        )
        
        if not result.get("success", False):
            raise Exception(result.get("error", "Unknown processing error"))
        
        # Get the user's RAG agent
        agent = get_user_agent(current_user)
        
        # Add processed chunks to the user's RAG index
        chunks = result.get("chunks", [])
        agent.add_audio_session(chunks)
        
        # If the user has a Letta agent, update its knowledge too
        if current_user.letta_agent_id:
            # Create a summary for the Letta agent
            summary = f"Audio Session Summary:\n\n"
            for chunk in chunks[:10]:  # Limit to first 10 chunks
                summary += f"- {chunk.get('summary', '')}\n"
            
            # Update the Letta agent's knowledge
            letta_mgr.update_agent_knowledge(
                current_user.letta_agent_id,
                summary,
                f"Audio Session {result.get('session_id')}"
            )
        
        return {
            "status": "success", 
            "session_id": result.get("session_id"),
            "segments_processed": len(chunks),
            "user_id": current_user.id
        }
        
    except Exception as e:
        error_detail = f"Audio processing failed: {str(e)}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-audio-json")
async def load_audio_json(
    file_path: str,
    current_user: User = Depends(get_current_user)
):
    """
    Load audio analysis JSON data into the user's knowledge base
    
    Args:
        file_path: Path to the JSON file
        current_user: Current authenticated user
        
    Returns:
        Loading status
    """
    try:
        logger.info(f"Loading audio JSON data for user {current_user.id}: {file_path}")
        
        # Validate the file path
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")
        
        # Get the user's RAG agent
        agent = get_user_agent(current_user)
        
        # Load the audio data into the user's RAG system
        agent.load_audio_data(file_path)
        
        return {
            "status": "success", 
            "message": f"Successfully loaded audio data from {file_path}",
            "user_id": current_user.id
        }
        
    except Exception as e:
        error_detail = f"Failed to load audio JSON: {str(e)}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-documents")
async def load_documents(
    directory: str = "therapy_documents",
    is_shared: bool = True,
    current_user: User = Depends(get_current_user)
):
    """
    Load documents into either the shared knowledge base or the user's personal knowledge
    
    Args:
        directory: Directory containing the documents
        is_shared: Whether the documents should be shared across all users
        current_user: Current authenticated user
        
    Returns:
        Loading status
    """
    try:
        # If loading shared documents, require admin privileges
        if is_shared and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can load shared documents"
            )
        
        logger.info(f"Loading documents for user {current_user.id} from {directory} (shared: {is_shared})")
        
        # Get the user's combined memory connector
        memory_connector = get_user_memory(current_user)
        
        # Handle based on whether we're loading shared documents or user-specific transcripts
        if is_shared:
            # For shared documents, reload the therapy knowledge index
            success = memory_connector.reload_therapy_knowledge()
            message = "Shared therapy documents reloaded"
        else:
            # For user-specific documents, reload the transcript memory
            success = memory_connector.reload_transcript_memory()
            message = f"User-specific transcripts reloaded for {current_user.id}"
        
        if not success:
            return {
                "status": "error",
                "message": f"Failed to load {'shared documents' if is_shared else 'user transcripts'}",
                "user_id": current_user.id
            }
        
        return {
            "status": "success",
            "message": f"Documents loaded successfully from {directory}",
            "user_id": current_user.id
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_detail = f"Error loading documents: {str(e)}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)
