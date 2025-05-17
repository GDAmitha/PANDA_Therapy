#!/usr/bin/env python3
"""
Vectorize Natey's Transcript

Simple script to vectorize Natey's transcript for memory retrieval.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

# Load environment variables for OpenAI API key
from dotenv import load_dotenv
load_dotenv()

# Import OpenAI embeddings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

def setup_openai_embeddings():
    """Configure OpenAI embeddings for vectorization"""
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.embed_model = embed_model
    Settings.llm = None  # Explicitly disable LLM component
    
def vectorize_natey_transcript():
    """Vectorize Natey's transcript using OpenAI embeddings"""
    try:
        # Import here after setting up OpenAI embeddings
        from backend.transcript_memory import TherapyTranscriptMemory
        
        # Set up OpenAI embeddings first
        setup_openai_embeddings()
        
        # Vectorize for Natey
        client_name = "natey"
        logger.info(f"Vectorizing transcript for client: {client_name}")
        
        memory = TherapyTranscriptMemory(client_name)
        success = memory.create_memory_index(force_reload=True)
        
        if success:
            logger.info(f"Successfully vectorized transcript for {client_name}")
            return True
        else:
            logger.error(f"Failed to vectorize transcript for {client_name}")
            return False
            
    except Exception as e:
        logger.error(f"Error vectorizing transcript: {e}")
        return False
        
if __name__ == "__main__":
    success = vectorize_natey_transcript()
    if success:
        print("✅ Successfully vectorized Natey's transcript")
        print("  Memory index created with OpenAI embeddings")
        print("  Ready for testing with combined_memory_connector.py")
    else:
        print("❌ Failed to vectorize Natey's transcript")
        print("  Check the error logs for details")
