"""Main FastAPI application for PANDA Therapy

This module serves as the entry point for the PANDA Therapy API, providing
multi-user support, authentication, and integration with personalized
Letta agents for each client.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dotenv import load_dotenv
import os
import logging
import traceback

# Import routes
from user_routes import router as user_router
from chat_routes import router as chat_router

# Import models and components
# Import our simple development authentication system
from simple_auth import get_current_user, create_dev_user
from models.user import User
from database import Database
from openai_llama_rag import OpenAITherapyIndex

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="PANDA Therapy API",
    description="API for the PANDA Therapy application with multi-user support and personalized agents",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database initialization
db = Database()

# No OAuth2 scheme needed for simple authentication

# Shared therapy knowledge RAG agent
shared_agent = None

# Include routers
app.include_router(user_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

# Simple dev authentication endpoint
@app.post("/api/dev-login")
async def dev_login(username: str, name: str, role: str = "patient"):
    """
    Simple development login that creates a user if needed
    Returns the user ID to be used in the X-User-ID header
    """
    user = create_dev_user(username, name, role)
    
    return {
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "message": "Use this user_id in the X-User-ID header for authentication"
    }

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running
    """
    return {"message": "PANDA Therapy API is running"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is functioning properly
    """
    return {"status": "ok", "version": "1.0.0"}

# Protected endpoint example
@app.get("/api/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    """
    Example protected route that requires authentication
    """
    return {
        "message": f"Hello, {current_user.name}!",
        "user_id": current_user.id,
        "role": current_user.role
    }

# Initialize a shared therapy knowledge index for all users
def init_shared_agent():
    """
    Initialize the shared therapy knowledge index using OpenAITherapyIndex
    """
    global shared_agent
    try:
        # Create the shared therapy index
        shared_agent = OpenAITherapyIndex()
        
        # Load shared therapy documents
        success = shared_agent.load_documents()
        
        if success:
            logger.info("Shared therapy knowledge index initialized successfully")
        else:
            logger.warning("Shared therapy knowledge index initialization had issues")
        
        return success
    except Exception as e:
        logger.error(f"Failed to initialize shared therapy index: {str(e)}\n{traceback.format_exc()}")
        return False

# Server startup event to initialize components
@app.on_event("startup")
async def startup_event():
    """
    Initialize necessary components on server startup
    """
    try:
        # Initialize user database directory
        user_data_dir = "./user_data"
        os.makedirs(user_data_dir, exist_ok=True)
        
        # Initialize shared storage directories
        shared_storage_dir = "./storage/shared"
        os.makedirs(os.path.join(shared_storage_dir, "documents"), exist_ok=True)
        
        # Initialize shared knowledge agent
        init_shared_agent()
        
        # Check for existing audio emotion files to potentially load
        audio_emo_dir = "backend/Audio/audio_emo_transcript"
        if os.path.exists(audio_emo_dir):
            logger.info(f"Found audio emotion directory at {audio_emo_dir}")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
