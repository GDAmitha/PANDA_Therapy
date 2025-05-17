"""
RAG Compatibility Layer for PANDA Therapy

This module provides a compatibility layer for the RAG system to handle
cases where external dependencies like Pinecone aren't properly installed.
"""

import os
import logging
import time
from typing import Optional, Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag to indicate if full RAG is available
full_rag_available = False
pinecone_available = False

# Try to import Pinecone and LlamaIndex components
try:
    # Try importing Pinecone directly to check availability
    try:
        import pinecone
        pinecone_available = True
        logger.info("Pinecone is available")
    except Exception as e:
        logger.warning(f"Pinecone is not available: {e}")
        pinecone_available = False
    
    # Check for LlamaIndex components
    try:
        from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
        from llama_index.core.node_parser import SimpleNodeParser
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core import ServiceContext, StorageContext
        full_rag_available = True
        logger.info("Full RAG system is available")
    except Exception as e:
        logger.warning(f"Full RAG system is not available: {e}")
        full_rag_available = False
        
except Exception as e:
    logger.warning(f"RAG component unavailable: {e}")
    full_rag_available = False


class SimpleFallbackIndex:
    """
    A simple fallback index that uses basic keyword matching when vector search isn't available
    """
    
    def __init__(self):
        """Initialize the fallback index"""
        self.documents = []
        self.initialized = False
        
    def load_documents(self, directory: str = None) -> bool:
        """Load documents from a directory using simple text loading"""
        try:
            if not directory:
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                directory = os.path.join(base_path, "backend", "therapy_documents")
                
            if not os.path.exists(directory):
                logger.warning(f"Document directory not found: {directory}")
                return False
                
            # Load all documents in the directory
            self.documents = []
            for filename in os.listdir(directory):
                if filename.endswith(".txt"):
                    file_path = os.path.join(directory, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            self.documents.append({
                                "content": content,
                                "source": file_path
                            })
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.documents)} documents in fallback mode")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading documents in fallback mode: {e}")
            return False
            
    def query(self, query_text: str) -> str:
        """
        Perform a simple keyword-based search
        """
        if not self.initialized or not self.documents:
            if not self.load_documents():
                return "Therapy knowledge is not available."
                
        # Very basic keyword search
        query_words = query_text.lower().split()
        results = []
        
        for doc in self.documents:
            content = doc["content"].lower()
            score = sum(1 for word in query_words if word in content)
            if score > 0:
                results.append({
                    "content": doc["content"],
                    "score": score,
                    "source": doc["source"]
                })
                
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top 3 results
        if not results:
            return "No relevant information found."
            
        output = "THERAPY KNOWLEDGE:\n\n"
        for i, result in enumerate(results[:3]):
            source = os.path.basename(result["source"])
            snippet = result["content"]
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            output += f"From {source}:\n{snippet}\n\n"
            
        return output
        

class SimpleFallbackTranscriptMemory:
    """
    A simple fallback transcript memory system that uses basic keyword matching
    """
    
    def __init__(self, user_id: str):
        """Initialize the fallback transcript memory"""
        self.user_id = user_id
        self.transcripts = []
        self.initialized = False
        self.memory_dir = None
        
    def create_memory_index(self) -> bool:
        """Load transcripts for a user using simple text loading"""
        try:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.memory_dir = os.path.join(base_path, "backend", "user_memories", f"user_{self.user_id}")
            
            if not os.path.exists(self.memory_dir):
                logger.warning(f"User memory directory not found: {self.memory_dir}")
                os.makedirs(self.memory_dir, exist_ok=True)
                self.initialized = True
                return True
                
            # Load all transcript files
            self.transcripts = []
            transcript_file = os.path.join(self.memory_dir, "transcripts.json")
            if os.path.exists(transcript_file):
                try:
                    import json
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        self.transcripts = json.load(f)
                        logger.info(f"Loaded {len(self.transcripts)} transcripts for user {self.user_id}")
                except Exception as e:
                    logger.error(f"Error reading transcripts for user {self.user_id}: {e}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading transcripts in fallback mode: {e}")
            return False
            
    def extract_memories_for_letta(self, query_text: str) -> str:
        """
        Extract relevant transcript memories using simple keyword matching
        """
        if not self.initialized:
            if not self.create_memory_index():
                return ""
                
        if not self.transcripts:
            return ""
                
        # Very basic keyword search
        query_words = query_text.lower().split()
        results = []
        
        for transcript in self.transcripts:
            # Get full transcript text
            full_text = ""
            for turn in transcript.get("transcript", []):
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                full_text += f"{speaker}: {text}\n"
                
            # Score the transcript
            content = full_text.lower()
            score = sum(1 for word in query_words if word in content)
            if score > 0:
                results.append({
                    "content": full_text,
                    "score": score,
                    "date": transcript.get("date", "")
                })
                
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top 2 results
        if not results:
            return ""
            
        output = ""
        for i, result in enumerate(results[:2]):
            date = result.get("date", "Previous session")
            snippet = result["content"]
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            output += f"From session {date}:\n{snippet}\n\n"
            
        return output


# Factory functions to get the appropriate implementation
def get_therapy_index():
    """Get the appropriate therapy index based on available components"""
    if full_rag_available:
        from openai_llama_rag import OpenAITherapyIndex
        return OpenAITherapyIndex()
    else:
        return SimpleFallbackIndex()
        
def get_transcript_memory(user_id: str):
    """Get the appropriate transcript memory based on available components"""
    if full_rag_available:
        from transcript_memory import TherapyTranscriptMemory
        return TherapyTranscriptMemory(user_id)
    else:
        return SimpleFallbackTranscriptMemory(user_id)
