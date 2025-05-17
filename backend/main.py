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
from backend.user_routes import router as user_router
from backend.chat_routes import router as chat_router
from backend.audio_routes import router as audio_router

# Import models and components
# Import our simple development authentication system
from backend.simple_auth import get_current_user, create_dev_user
from backend.models.user import User
from backend.database import Database
from backend.openai_llama_rag import OpenAITherapyIndex

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

# Add this global variable
DEFAULT_USER_ID = None

# Include routers
app.include_router(user_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(audio_router, prefix="/api")

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

# Add endpoint to get the default user ID
@app.get("/api/default-user")
async def get_default_user():
    """Get the default user ID for use in the frontend"""
    if DEFAULT_USER_ID:
        return {"user_id": DEFAULT_USER_ID}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No default user available"
        )

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

# Add this function to create a default user
def create_default_user():
    global DEFAULT_USER_ID
    try:
        # Create a default user
        default_user = create_dev_user(
            username="default",
            name="Default User",
            role="patient"
        )
        DEFAULT_USER_ID = default_user.id
        
        # Store in environment variable for easy access
        os.environ["DEFAULT_USER_ID"] = DEFAULT_USER_ID
        
        logger.info(f"Created default user with ID: {DEFAULT_USER_ID}")
        
        # Optional: Create a Letta agent for this user if needed
        # This is commented out since letta_manager isn't imported in this file
        # If you need Letta integration, uncomment and import letta_manager
        # try:
        #     from backend.letta_agent import LettaAgentManager
        #     letta_manager = LettaAgentManager()
        #     agent_id = letta_manager.create_agent_for_user(default_user)
        #     if agent_id:
        #         logger.info(f"Created Letta agent {agent_id} for default user")
        # except Exception as e:
        #     logger.warning(f"Could not create Letta agent for default user: {e}")
    except Exception as e:
        logger.error(f"Failed to create default user: {e}")

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
        
        # Create default user
        create_default_user()
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
