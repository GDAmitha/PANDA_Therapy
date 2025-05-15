# rag_agent.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, CompositeIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexDict
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core import Document, StorageContext, load_index_from_storage
import os
import json
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class TherapyRAGAgent:
    def __init__(self):
        try:
            # Initialize OpenAI components
            self.embed_model = OpenAIEmbedding()
            self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
            
            # Configure global settings
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            
            # Initialize memory
            self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            
            # Initialize separate indices for documents and audio data
            self.doc_index = None
            self.audio_index = None
            self.composite_index = None
            
            # Storage directories
            self.doc_storage_dir = "./storage/documents"
            self.audio_storage_dir = "./storage/audio"
            
            # Create storage directories if they don't exist
            os.makedirs(self.doc_storage_dir, exist_ok=True)
            os.makedirs(self.audio_storage_dir, exist_ok=True)
            
            # Try to load existing indices
            self._load_existing_indices()
            
        except Exception as e:
            logger.error(f"Error initializing RAG agent: {str(e)}")
            raise
    
    def _load_existing_indices(self):
        """Attempt to load existing indices from storage."""
        try:
            indices_loaded = False
            
            # Try to load document index
            if os.path.exists(self.doc_storage_dir) and os.listdir(self.doc_storage_dir):
                logger.info("Loading existing document index...")
                storage_context = StorageContext.from_defaults(persist_dir=self.doc_storage_dir)
                self.doc_index = load_index_from_storage(storage_context)
                indices_loaded = True
            
            # Try to load audio index
            if os.path.exists(self.audio_storage_dir) and os.listdir(self.audio_storage_dir):
                logger.info("Loading existing audio index...")
                storage_context = StorageContext.from_defaults(persist_dir=self.audio_storage_dir)
                self.audio_index = load_index_from_storage(storage_context)
                indices_loaded = True
            
            # Create composite index if any indices were loaded
            if indices_loaded:
                self._create_composite_index()
        
        except Exception as e:
            logger.error(f"Error loading existing indices: {str(e)}")
            # Continue without existing indices
            pass
        
    def _create_composite_index(self):
        """Create a composite index from available indices."""
        indices_dict = {}
        
        if self.doc_index:
            indices_dict["documents"] = self.doc_index
        
        if self.audio_index:
            indices_dict["audio"] = self.audio_index
        
        if indices_dict:
            # Create composite index if we have at least one sub-index
            self.composite_index = CompositeIndex(indices_dict)
            logger.info("Created composite index")
        
    def load_documents(self, directory_path: str):
        """Load and index documents from a directory."""
        try:
            logger.info(f"Loading documents from {directory_path}")
            
            # Load and parse documents
            documents = SimpleDirectoryReader(
                directory_path,
                recursive=True,
                required_exts=[".txt"]
            ).load_data()
            
            if not documents:
                logger.warning(f"No documents found in {directory_path}")
                return
            
            # Create document index
            self.doc_index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
            
            # Persist index
            self.doc_index.storage_context.persist(persist_dir=self.doc_storage_dir)
            logger.info(f"Indexed {len(documents)} documents successfully")
            
            # Update composite index
            self._create_composite_index()
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def load_audio_data(self, audio_data_path: str):
        """
        Load and index audio emotion analysis data.
        
        Args:
            audio_data_path: Path to the JSON file containing audio emotion analysis
        """
        try:
            logger.info(f"Loading audio data from {audio_data_path}")
            
            # Load emotion analysis data
            with open(audio_data_path, 'r') as f:
                data = json.load(f)
            
            emotion_analysis = data.get("emotion_analysis", [])
            if not emotion_analysis:
                logger.warning("No emotion analysis data found")
                return
            
            # Convert to documents
            documents = []
            audio_file = data.get("audio_file", "unknown_audio")
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            
            for i, item in enumerate(emotion_analysis):
                # Create a summary for the document text
                text = item.get("text", "")
                speaker = item.get("speaker", "Unknown")
                wav_emotion = item.get("predicted_wav_emotion", "neutral")
                text_emotion = item.get("predicted_text_emotion", "neutral")
                
                # Create a summary that includes all information
                summary = (
                    f"{speaker} said: \"{text}\" "
                    f"Voice emotion: {wav_emotion} "
                    f"Text emotion: {text_emotion}"
                )
                
                # Create the document with metadata
                doc = Document(
                    text=summary,
                    metadata={
                        "id": f"{base_name}_chunk_{i}",
                        "speaker": speaker,
                        "emotion": wav_emotion,
                        "text_emotion": text_emotion,
                        "source": "audio_session",
                        "timestamp": item.get("start", 0),
                        "audio_file": audio_file
                    }
                )
                
                documents.append(doc)
            
            # Create or update audio index
            if self.audio_index is None:
                # First time - create a new index
                self.audio_index = VectorStoreIndex.from_documents(documents)
            else:
                # Add to existing index
                for doc in documents:
                    self.audio_index.insert(doc)
            
            # Persist audio index
            self.audio_index.storage_context.persist(persist_dir=self.audio_storage_dir)
            logger.info(f"Indexed {len(documents)} audio segments successfully")
            
            # Update composite index
            self._create_composite_index()
            
        except Exception as e:
            logger.error(f"Error loading audio data: {str(e)}")
            raise

    def chat(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None):
        """Process a chat query using the available indices."""
        try:
            # Determine which index to use
            index_to_use = self.composite_index or self.doc_index or self.audio_index
            
            if not index_to_use:
                logger.warning("No indices initialized")
                return "Please load documents or audio data first."
            
            # Create chat engine with context
            chat_engine = ContextChatEngine.from_defaults(
                index=index_to_use,
                memory=self.memory,
                system_prompt=(
                    "You are a helpful therapy assistant. Use the provided context to answer "
                    "questions about therapy and mental health. If the context includes audio "
                    "session data with emotions, consider how the emotional state might affect "
                    "your response."
                ),
                similarity_top_k=5
            )
            
            # Format chat history for memory
            if chat_history:
                for message in chat_history:
                    if message.get("role") == "user":
                        self.memory.put({"role": "user", "content": message["content"]})
                    elif message.get("role") == "assistant":
                        self.memory.put({"role": "assistant", "content": message["content"]})
            
            # Get response
            response = chat_engine.chat(query)
            return response.response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise

    def add_chunk(self, chunk: Dict[str, Any], is_audio: bool = False):
        """
        Append a single chunk to the appropriate index.
        
        Args:
            chunk: Dictionary containing chunk data
            is_audio: Whether this is audio data (True) or document data (False)
        """
        try:
            # Create document with metadata
            doc = Document(
                text=chunk.get("summary", ""),
                metadata={
                    "id": chunk.get("id", ""),
                    "emotion": chunk.get("emotion", ""),
                    "text_emotion": chunk.get("text_emotion", ""),
                    "source": chunk.get("source", ""),
                    "speaker": chunk.get("speaker", ""),
                    "timestamp": chunk.get("timestamp", 0),
                    "audio_file": chunk.get("audio_file", "") if is_audio else ""
                }
            )
            
            if is_audio:
                # Add to audio index
                if self.audio_index is None:
                    # First time - create a new index
                    self.audio_index = VectorStoreIndex.from_documents([doc])
                else:
                    # Add to existing index
                    self.audio_index.insert(doc)
                
                # Persist changes
                self.audio_index.storage_context.persist(persist_dir=self.audio_storage_dir)
                logger.info("Successfully added new audio chunk to index")
            else:
                # Add to document index
                if self.doc_index is None:
                    # First time - create a new index
                    self.doc_index = VectorStoreIndex.from_documents([doc])
                else:
                    # Add to existing index
                    self.doc_index.insert(doc)
                
                # Persist changes
                self.doc_index.storage_context.persist(persist_dir=self.doc_storage_dir)
                logger.info("Successfully added new document chunk to index")
            
            # Update composite index
            self._create_composite_index()
            
        except Exception as e:
            logger.error(f"Error adding chunk: {str(e)}")
            raise

    def add_audio_session(self, chunks: List[Dict[str, Any]]):
        """
        Add all chunks from an audio session to the audio index.
        
        Args:
            chunks: List of chunks from audio processing
        """
        try:
            logger.info(f"Adding {len(chunks)} audio chunks to index")
            
            for chunk in chunks:
                self.add_chunk(chunk, is_audio=True)
                
            logger.info("Successfully added all audio chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding audio session: {str(e)}")
            return False

# Usage example
if __name__ == "__main__":
    agent = TherapyRAGAgent()
    agent.load_documents("./therapy_documents")
    response = agent.chat("How can I manage anxiety?")
    print(response)