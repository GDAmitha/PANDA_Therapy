# rag_agent.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core import Document
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

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
            
            # Initialize memory and index
            self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            self.index = None
            
            # Try to load existing index
            if os.path.exists("./storage"):
                logger.info("Loading existing index...")
                self.index = VectorStoreIndex.from_documents(
                    SimpleDirectoryReader("./therapy_documents").load_data(),
                    show_progress=True
                )
                self.index.storage_context.persist(persist_dir="./storage")
        except Exception as e:
            logger.error(f"Error initializing RAG agent: {str(e)}")
            raise
        
    def load_documents(self, directory_path: str):
        try:
            logger.info(f"Loading documents from {directory_path}")
            
            # Load and parse documents
            documents = SimpleDirectoryReader(
                directory_path,
                recursive=True,
                required_exts=[".txt"]
            ).load_data()
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
            
            # Persist index
            self.index.storage_context.persist(persist_dir="./storage")
            logger.info("Documents loaded and indexed successfully")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def chat(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None):
        try:
            if not self.index:
                logger.warning("Index not initialized")
                return "Please load documents first."
            
            # Create chat engine with context
            chat_engine = ContextChatEngine.from_defaults(
                index=self.index,
                memory=self.memory,
                system_prompt="You are a helpful therapy assistant. Use the provided context to answer questions about therapy and mental health.",
                similarity_top_k=3
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

    def add_chunk(self, chunk: Dict[str, Any]):
        """
        Append a single transcript/emotion chunk to the index.
        Expects output of generate_rag_json().
        """
        try:
            
            
            # Create document with metadata
            doc = Document(
                text=chunk["summary"],
                metadata={
                    "id": chunk["id"],
                    "emotion": chunk["emotion"],
                    "source": chunk["source"],
                    "speaker": chunk["speaker"],
                    "timestamp": chunk["timestamp"],
                }
            )
            
            if self.index is None:
                # First time - create a new index
                self.index = VectorStoreIndex.from_documents([doc])
            else:
                # Add to existing index
                self.index.insert(doc)
            
            # Persist changes
            self.index.storage_context.persist(persist_dir="./storage")
            logger.info("Successfully added new chunk to index")
            
        except Exception as e:
            logger.error(f"Error adding chunk: {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    agent = TherapyRAGAgent()
    agent.load_documents("./therapy_documents")
    response = agent.chat("How can I manage anxiety?")
    print(response)