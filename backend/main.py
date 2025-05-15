from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from rag_agent import TherapyRAGAgent
from audio_processor import AudioProcessor
import shutil, tempfile
import logging
import traceback
import json
import os

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

# Initialize the RAG agent and audio processor
try:
    agent = TherapyRAGAgent()
    audio_processor = AudioProcessor()
    logger.info("RAG agent and audio processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}\n{traceback.format_exc()}")
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

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...), therapist_name: str = Form("Therapist"), patient_name: str = Form("Patient")):
    try:
        logger.info(f"Processing audio file: {file.filename}")
        
        # 1) Save to a temporary file
        suffix = ".wav" if file.filename.endswith(".wav") else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        logger.info(f"Saved audio to temporary file: {tmp_path}")
        
        # 2) Process audio file through the pipeline
        result = audio_processor.process_audio_session(
            tmp_path, 
            therapist_name=therapist_name,
            patient_name=patient_name
        )
        
        if not result.get("success", False):
            raise Exception(result.get("error", "Unknown processing error"))
        
        # 3) Add processed chunks to the RAG system
        chunks = result.get("chunks", [])
        agent.add_audio_session(chunks)
        
        return {
            "status": "success", 
            "session_id": result.get("session_id"),
            "segments_processed": len(chunks)
        }
    except Exception as e:
        error_detail = f"Audio processing failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-audio-json")
async def load_audio_json(file_path: str):
    try:
        logger.info(f"Loading audio JSON data from: {file_path}")
        
        # Validate the file path
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")
        
        # Load into RAG system
        agent.load_audio_data(file_path)
        
        return {"status": "success", "message": f"Successfully loaded audio data from {file_path}"}
    except Exception as e:
        error_detail = f"Failed to load audio JSON: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    try:
        # Load documents on startup
        logger.info("Loading documents on startup...")
        agent.load_documents("therapy_documents")
        logger.info("Initial document loading successful")
        
        # Check for existing audio emotion files to load
        audio_emo_dir = "backend/Audio/audio_emo_transcript"
        if os.path.exists(audio_emo_dir):
            for filename in os.listdir(audio_emo_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(audio_emo_dir, filename)
                    logger.info(f"Loading existing audio emotion file: {file_path}")
                    agent.load_audio_data(file_path)
    except Exception as e:
        logger.error(f"Failed to load initial data on startup: {str(e)}\n{traceback.format_exc()}")
        # Continue running even if initial loading fails
        pass

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug") 