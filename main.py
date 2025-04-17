from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rag_agent import TherapyRAGAgent
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG agent
try:
    agent = TherapyRAGAgent()
    logger.info("RAG agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG agent: {str(e)}\n{traceback.format_exc()}")
    raise

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[dict]] = []

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request with message: {request.message}")
        logger.info(f"Chat history: {request.chat_history}")
        
        # Validate chat_history format
        if not isinstance(request.chat_history, list):
            logger.error("Chat history is not a list")
            raise HTTPException(status_code=400, detail="Chat history must be a list")
            
        response = agent.chat(request.message, request.chat_history)
        logger.info("Successfully generated response")
        return {"response": response}
    except Exception as e:
        error_detail = f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/load-documents")
async def load_documents(directory: str = "therapy_documents"):
    try:
        logger.info(f"Loading documents from directory: {directory}")
        agent.load_documents(directory)
        logger.info("Documents loaded successfully")
        return {"status": "success"}
    except Exception as e:
        error_detail = f"Error loading documents: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

if __name__ == "__main__":
    import uvicorn
    try:
        # Load documents on startup
        logger.info("Loading documents on startup...")
        agent.load_documents("therapy_documents")
        logger.info("Initial document loading successful")
    except Exception as e:
        logger.error(f"Failed to load documents on startup: {str(e)}\n{traceback.format_exc()}")
        raise

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug") 