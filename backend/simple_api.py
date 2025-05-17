"""
Simple FastAPI server with multi-user support for PANDA Therapy

This is a simplified version that demonstrates the multi-user architecture
without requiring complex dependencies like LlamaIndex.
"""

from fastapi import FastAPI, Depends, HTTPException, status, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import os
import json
import uuid
import shutil
import tempfile
import logging
import time
from datetime import datetime
from pydantic import BaseModel, Field

# Load environment variables from .env file
from dotenv import load_dotenv
# Look for .env file in parent directory since we're in the backend folder
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment variables from {env_path}")
    print(f"LETTA_API_KEY is {'set' if os.getenv('LETTA_API_KEY') else 'not set'}")
else:
    print(f"Warning: .env file not found at {env_path}")


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LettaAgentManager
try:
    from letta_agent import LettaAgentManager
    letta_available = True
    logger.info("Letta agent integration available")
except ImportError:
    letta_available = False
    logger.warning("Letta agent integration not available - using mock responses")

# Create FastAPI app
app = FastAPI(title="PANDA Therapy Simple Multi-User API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and RAG system
from database import Database
db = Database()

# In-memory caches for active sessions
user_chats = {}
user_audio = {}
user_letta_agents = {}

# Initialize Letta agent manager if available
letta_manager = None
if letta_available:
    try:
        # Initialize with the API key from environment variable
        letta_manager = LettaAgentManager()
        logger.info("Letta agent manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Letta agent manager: {str(e)}")
        
# Set up to use our compatibility layer
from rag_compatibility import full_rag_available

# Import the memory connector with fallback mode
from combined_memory_connector import CombinedMemoryConnector, rag_available

# Always keep rag_available True since we have fallbacks
if rag_available:
    logger.info("Memory system loaded successfully")
    if full_rag_available:
        logger.info("Using full RAG capabilities with vector search")
    else:
        logger.info("Using fallback RAG capabilities with keyword search")

# Create required directories for the memory system
try:
    # Create required directories if they don't exist
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "therapy_documents"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_memories"), exist_ok=True)
    
    logger.info("Memory system directories created")
except Exception as e:
    logger.error(f"Error setting up memory system: {e}")
    rag_available = False

# Try to import memory connector components
try:
    # We'll initialize memory connectors per user
    user_memory_connectors = {}
    logger.info("Memory connector system initialized successfully")
except Exception as e:
    rag_available = False
    logger.warning(f"Memory system not available - {str(e)}")
    logger.warning("Using simplified responses without knowledge retrieval capabilities")

# Data classes
class UserBase(BaseModel):
    username: str
    name: str
    role: str = "patient"

class User(UserBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    letta_agent_id: Optional[str] = None
    transcripts: List[Dict[str, Any]] = []

class ChatMessage(BaseModel):
    message: str
    chat_history: Optional[List[Dict[str, Any]]] = []

class AudioData(BaseModel):
    transcript: List[Dict[str, Any]]
    session_id: Optional[str] = None
    therapist: str = "Therapist"
    patient: str = "Patient"

# User management
async def get_current_user(x_user_id: Optional[str] = Header(None)):
    """Simple user authentication via header using persistent database"""
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-ID header is required for authentication",
            headers={"WWW-Authenticate": "XUserID"},
        )
    
    # Get the user from the database
    user = db.get_user(x_user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID",
            headers={"WWW-Authenticate": "XUserID"},
        )
    
    return user

# Routes
@app.post("/api/dev-login")
async def create_user(username: str, name: str, role: str = "patient"):
    """Create a new user or get existing one by username using persistent database"""
    # Check if username exists
    existing_user = db.get_user_by_username(username)
    if existing_user:
        # Return existing user info
        return {
            "user_id": existing_user.id,
            "username": existing_user.username,
            "name": existing_user.name,
            "role": existing_user.role,
            "message": "Existing user. Use this user_id in X-User-ID header for authentication."
        }
    
    # If user doesn't exist, create a new one
    # For development, we can create users without passwords
    # In a production app, you would require and hash passwords
    from models.user import UserRole
    role_enum = UserRole(role) if role in [r.value for r in UserRole] else UserRole.PATIENT
    
    # Create user in the database with minimal info for development
    new_user = db.create_user(
        username=username,
        email=f"{username}@example.com",  # Placeholder email for development
        name=name,
        password="devpassword",  # Simple password for development
        role=role_enum
    )
    
    if not new_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
    
    # Initialize empty caches for this user
    user_chats[new_user.id] = []
    user_audio[new_user.id] = []
    
    # Create a Letta agent for this user if available
    if letta_manager and letta_available:
        try:
            logger.info(f"Creating Letta agent for new user {username}")
            agent_id = letta_manager.create_agent_for_user(new_user)
            
            if agent_id:
                logger.info(f"Created Letta agent {agent_id} for user {username}")
                # Store the agent ID in both memory cache and the database
                user_letta_agents[new_user.id] = agent_id
                db.update_user_letta_agent(new_user.id, agent_id)
            else:
                logger.error(f"Failed to create Letta agent for user {username}")
        except Exception as e:
            logger.error(f"Failed to create Letta agent for user {username}: {str(e)}")
    
    return {
        "user_id": new_user.id,
        "username": new_user.username,
        "name": new_user.name,
        "role": new_user.role,
        "message": "New user created. Use this user_id in X-User-ID header for authentication."
    }

@app.get("/api/users/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "name": current_user.name,
        "role": current_user.role,
        "transcript_count": len(current_user.transcripts)
    }

@app.post("/api/chat/message")
async def chat_message(
    chat_request: ChatMessage, 
    current_user: User = Depends(get_current_user)
):
    """Process a chat message with multi-user isolation, Letta and RAG integration"""
    # Store the message in user history (in-memory for the session)
    if current_user.id not in user_chats:
        user_chats[current_user.id] = []
        
    user_chats[current_user.id].append({
        "role": "user", 
        "content": chat_request.message
    })

    # Check if we should use combined memory to augment the message with relevant context
    therapy_knowledge = None
    transcript_memory = None
    combined_context = None
    
    # Get or create memory connector for this user
    if rag_available:
        try:
            # Get or initialize the user's memory connector
            if current_user.id not in user_memory_connectors:
                user_memory_connectors[current_user.id] = CombinedMemoryConnector(current_user.id)
                logger.info(f"Created new memory connector for user {current_user.username}")
            
            memory_connector = user_memory_connectors[current_user.id]
            
            # Get relevant content from both general therapy documents and patient-specific transcripts
            start_time = time.time()
            therapy_knowledge = memory_connector.get_therapy_knowledge(chat_request.message)
            therapy_time = time.time() - start_time
            
            start_time = time.time()
            transcript_memory = memory_connector.get_transcript_memory(chat_request.message)
            transcript_time = time.time() - start_time
            
            # Log detailed information about what was retrieved
            logger.info(f"RETRIEVAL METRICS for user {current_user.username} query: '{chat_request.message[:30]}...'")
            logger.info(f"  - Therapy knowledge retrieval time: {therapy_time:.2f}s")
            logger.info(f"  - Transcript memory retrieval time: {transcript_time:.2f}s")
            logger.info(f"  - Total retrieval time: {therapy_time + transcript_time:.2f}s")
            
            # Combine the context
            combined_context = ""
            if therapy_knowledge:
                # Truncate for logging but use full content for the agent
                therapy_log = therapy_knowledge[:150] + "..." if len(therapy_knowledge) > 150 else therapy_knowledge
                logger.info(f"THERAPY KNOWLEDGE RETRIEVED:\n{therapy_log}")
                combined_context += "THERAPY KNOWLEDGE:\n" + therapy_knowledge + "\n\n"
                
            if transcript_memory:
                # Truncate for logging but use full content for the agent
                transcript_log = transcript_memory[:150] + "..." if len(transcript_memory) > 150 else transcript_memory
                logger.info(f"PATIENT TRANSCRIPT MEMORY RETRIEVED:\n{transcript_log}")
                combined_context += "PATIENT TRANSCRIPT MEMORY:\n" + transcript_memory
                
            if combined_context.strip() == "":
                combined_context = None
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")

    # Use Letta agent if available
    if letta_manager and letta_available:
        try:
            # Get or assign a Letta agent for this user
            agent_id = None
            
            # Check if user has an agent ID stored in the database
            db_user = db.get_user(current_user.id)
            if db_user and hasattr(db_user, 'letta_agent_id') and db_user.letta_agent_id:
                agent_id = db_user.letta_agent_id
                user_letta_agents[current_user.id] = agent_id  # Cache it for the session
                logger.info(f"Using Letta agent {agent_id} from database for {current_user.username}")
            # Check session cache if not in database
            elif current_user.id in user_letta_agents:
                agent_id = user_letta_agents[current_user.id]
                logger.info(f"Using cached Letta agent {agent_id} for user {current_user.username}")
            # Create a new agent if none exists
            else:
                logger.info(f"No agent found for {current_user.username}, creating a new one")
                agent_id = letta_manager.create_agent_for_user(current_user)
                if agent_id:
                    # Store the agent ID in both session cache and database
                    user_letta_agents[current_user.id] = agent_id
                    db.update_user_letta_agent(current_user.id, agent_id)
                    logger.info(f"Created new Letta agent {agent_id} for user {current_user.username}")
                else:
                    raise Exception(f"Failed to create Letta agent for user {current_user.username}")
            
            # Format the message for Letta, including combined context if available
            enhanced_message = chat_request.message
            if combined_context:
                enhanced_message = f"[CONTEXT: {combined_context}]\n\nUSER: {chat_request.message}"
            else:
                enhanced_message = chat_request.message
            
            # Send the message to the Letta agent
            logger.info(f"Sending message to Letta agent {agent_id} for user {current_user.username}")
            result = letta_manager.chat_with_agent(agent_id, enhanced_message)
            
            # Extract the response from the result
            if "error" in result:
                response = f"I'm sorry, there was an error processing your request: {result['error']}"
                logger.error(f"Error from Letta agent: {result['error']}")
            else:
                response = result.get("response", "I'm sorry, I couldn't process your request.")
                logger.info(f"Received response from Letta agent for user {current_user.username}")
                
                # Log additional data from Letta if available
                if "sources" in result:
                    logger.info(f"Letta used sources in response: {result['sources']}")
                    
                # Prefix the response to clearly indicate it came from Letta
                if not response.startswith("[LETTA]"): 
                    response = f"[LETTA] {response}"
        except Exception as e:
            logger.error(f"Error communicating with Letta agent: {str(e)}")
            response = f"I'm having trouble accessing the therapy system right now. Please try again later."
    else:
        # If no Letta agent available or there was an error, generate a simulated response
        response_text = f"I understand you're asking about '{chat_request.message}'"

        # Add context from both therapy knowledge and transcript memory if available
        if therapy_knowledge:
            response_text += f"\n\nBased on therapy knowledge, I can tell you: {therapy_knowledge}"

        if transcript_memory:
            response_text += f"\n\nBased on our previous sessions, I can see: {transcript_memory}"

        response = response_text

    
    # Store the response in session cache
    user_chats[current_user.id].append({
        "role": "assistant",
        "content": response
    })
    
    return {"response": response}

@app.post("/api/audio/simulate")
async def simulate_audio(
    audio_data: AudioData,
    current_user: User = Depends(get_current_user)
):
    """Simulate audio processing - store transcript data in the database"""
    # Add a session ID if none provided
    if not audio_data.session_id:
        audio_data.session_id = str(uuid.uuid4())
    
    # Store the session in the database
    session_data = {
        "session_id": audio_data.session_id,
        "therapist": audio_data.therapist,
        "patient": audio_data.patient,
        "created_at": datetime.now().isoformat(),
    }
    
    session_id = db.create_therapy_session(current_user.id, session_data)
    
    # Store transcript in the database
    transcript_data = {
        "session_id": session_id,
        "transcript": audio_data.transcript,
        "emotion_analysis": extract_emotions(audio_data.transcript)
    }
    
    transcript_id = db.create_transcript(current_user.id, transcript_data)
    
    # Also keep in memory cache for quick access during this session
    if current_user.id not in user_audio:
        user_audio[current_user.id] = []
    
    user_audio[current_user.id].append(audio_data.dict())
    
    # If RAG system is available, add this transcript to the RAG index
    if rag_available and rag_agent:
        try:
            # Format transcript for RAG indexing
            rag_document = format_transcript_for_rag(audio_data.transcript, audio_data.session_id)
            rag_agent.add_document_for_user(current_user.id, rag_document, source_type="audio_transcript")
            logger.info(f"Added transcript to RAG index for user {current_user.id}")
        except Exception as e:
            logger.error(f"Failed to add transcript to RAG index: {str(e)}")
    
    return {
        "status": "success",
        "message": "Audio data processed and stored in database",
        "session_id": audio_data.session_id,
        "transcript_id": transcript_id,
        "transcript_count": len(audio_data.transcript)
    }

# Helper function to extract emotions from transcript
def extract_emotions(transcript):
    """Extract emotions from transcript for analysis"""
    emotions = []
    for entry in transcript:
        text = entry.get("text", "")
        speaker = entry.get("speaker", "")
        
        # Simple keyword-based emotion extraction for the demo
        # In production, this would use a proper NLP model like the one in your audio processing pipeline
        detected_emotions = []
        emotion_keywords = {
            "anxiety": ["anxious", "worried", "nervous", "fear", "stress"],
            "happiness": ["happy", "joy", "excited", "pleased", "glad"],
            "sadness": ["sad", "depressed", "down", "unhappy", "miserable"],
            "anger": ["angry", "frustrated", "annoyed", "irritated", "mad"],
            "surprise": ["surprised", "shocked", "astonished", "amazed"],
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                detected_emotions.append(emotion)
        
        if detected_emotions:
            emotions.append({
                "speaker": speaker,
                "text": text,
                "emotions": detected_emotions,
                "confidence": 0.8  # Placeholder confidence value
            })
    
    return emotions

# Helper function to format transcript for RAG
def format_transcript_for_rag(transcript, session_id):
    """Format transcript data for RAG indexing"""
    # Combine transcript entries into a structured document
    full_text = f"Therapy Session {session_id}\n\n"
    
    for entry in transcript:
        speaker = entry.get("speaker", "Unknown")
        text = entry.get("text", "")
        full_text += f"{speaker}: {text}\n"
    
    return full_text

# Import the audio processing components using the compatibility layer
audio_processing_available = False
therapy_analyzer = None

try:
    # Add the parent directory to path to ensure modules can be imported
    import sys
    import os
    # Get the absolute path of the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Make sure the current directory is in the path
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Import the compatibility layer which handles all the audio modules
    from Audio.audio_compat import is_audio_processing_available, initialize_therapy_analyzer, process_audio_file
    
    # Check if audio processing is available
    audio_processing_available = is_audio_processing_available()
    
    # Initialize the therapy analyzer
    therapy_analyzer = initialize_therapy_analyzer()
    
    if audio_processing_available and therapy_analyzer:
        logger.info("Audio processing and therapy analysis components loaded successfully")
    elif audio_processing_available:
        logger.info("Audio processing available but therapy analyzer could not be initialized")
    else:
        logger.warning("Audio processing components not available")
except ImportError as e:
    logger.warning(f"Failed to import audio compatibility layer: {str(e)}")
except Exception as e:
    logger.error(f"Error initializing audio processing: {str(e)}")

@app.post("/api/audio/upload")
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
            
            # Generate a session ID
            session_id = str(uuid.uuid4())
            
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
                    
                    # Analyze the therapy session with the user's Letta agent
                    analysis_result = therapy_analyzer.analyze_transcript_data(
                        current_user.id, 
                        transcript_data
                    )
                    
                    # Store the analysis in the database
                    if "analysis_summary" in analysis_result:
                        analysis_data = {
                            "session_id": db_session_id,
                            "transcript_id": transcript_id,
                            "summary": analysis_result["analysis_summary"],
                            "created_at": datetime.now().isoformat(),
                        }
                        analysis_id = db.create_session_analysis(current_user.id, analysis_data)
                    
                    # Also add to the RAG system if available
                    if rag_available and rag_agent:
                        try:
                            # Format for RAG indexing
                            rag_document = format_transcript_for_rag(transcript_data["transcript"], session_id)
                            
                            # Create document object for RAG indexing
                            from llama_index import Document
                            doc = Document(
                                text=rag_document,
                                metadata={
                                    "source": "audio_transcript",
                                    "session_id": session_id,
                                    "user_id": current_user.id
                                }
                            )
                            
                            # Make sure the user has an index
                            rag_agent.create_user_index_if_not_exists(current_user.id)
                            
                            # Add document to the user's index
                            rag_agent.add_user_documents(
                                user_id=current_user.id, 
                                documents=[doc]
                            )
                            logger.info(f"Added processed audio transcript to RAG index for user {current_user.id}")
                        except Exception as e:
                            logger.error(f"Failed to add transcript to RAG index: {str(e)}")
                    
                    return {
                        "status": "success",
                        "message": "Audio file processed with emotion analysis",
                        "session_id": session_id,
                        "transcript_id": transcript_id,
                        "analysis_id": analysis_id if "analysis_summary" in analysis_result else None,
                        "analysis_summary": analysis_result.get("analysis_summary", None),
                        "transcript_count": len(transcript_data["transcript"])
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing audio file: {str(e)}")
                    # Fall back to simulated processing on error
            
            # If audio processing is not available or failed, create a simulated transcript
            logger.warning("Using simulated transcript - audio processing not available")
            simulated_transcript = [
                {"speaker": therapist_name, "text": "Hello, how are you feeling today?"},
                {"speaker": patient_name, "text": "I'm feeling a bit anxious about work."},
                {"speaker": therapist_name, "text": "Can you tell me more about what's causing your anxiety?"},
                {"speaker": patient_name, "text": "I have a big presentation coming up and I'm worried I'll mess it up."},
                {"speaker": therapist_name, "text": "That's understandable. Let's talk about some strategies to help with presentation anxiety."},
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
                "emotion_analysis": extract_emotions(simulated_transcript),
                "simulated": True,
            }
            
            transcript_id = db.create_transcript(current_user.id, transcript_data)
            
            # Also keep in memory cache for quick access during this session
            if current_user.id not in user_audio:
                user_audio[current_user.id] = []
            
            audio_data = AudioData(
                transcript=simulated_transcript,
                session_id=session_id,
                therapist=therapist_name,
                patient=patient_name
            )
            
            user_audio[current_user.id].append(audio_data.dict())
            
            # Also add to the RAG system if available
            if rag_available and rag_agent:
                try:
                    # Format for RAG indexing
                    rag_document = format_transcript_for_rag(simulated_transcript, session_id)
                    
                    # Create document object for RAG indexing
                    from llama_index import Document
                    doc = Document(
                        text=rag_document,
                        metadata={
                            "source": "simulated_audio_transcript",
                            "session_id": session_id,
                            "user_id": current_user.id
                        }
                    )
                    
                    # Make sure the user has an index
                    rag_agent.create_user_index_if_not_exists(current_user.id)
                    
                    # Add document to the user's index
                    rag_agent.add_user_documents(
                        user_id=current_user.id, 
                        documents=[doc]
                    )
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
            os.unlink(temp_file.name)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
