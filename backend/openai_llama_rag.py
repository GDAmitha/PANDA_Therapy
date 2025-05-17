"""
LlamaIndex RAG System with OpenAI Embeddings

This module implements a RAG system using LlamaIndex with OpenAI embeddings
for more accurate document retrieval.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import Pinecone manager
from pinecone_store import pinecone_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path setup
BASE_PATH = Path(__file__).parent.parent.absolute()
DOCS_PATH = os.path.join(BASE_PATH, "therapy_documents")
INDEX_PERSIST_DIR = os.path.join(BASE_PATH, "backend", "openai_llama_index_storage")

class OpenAITherapyIndex:
    """
    Therapy document indexing system using LlamaIndex with OpenAI embeddings
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, cache_size: int = 50):
        """
        Initialize the OpenAI therapy indexing system
        
        Args:
            openai_api_key: Optional API key for OpenAI, will use env var if not provided
            cache_size: Size of the query cache (default: 50 entries)
        """
        # Simple LRU cache for query results
        self.query_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.index = None
        self.query_engine = None
        self.openai_api_key = openai_api_key
        
        # Set OpenAI API key if provided
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OpenAI API key is required")
            
        # Initialize Pinecone index name
        self.pinecone_index_name = None
        
    def load_documents(self, force_reload: bool = False) -> bool:
        """
        Load and index therapy documents with OpenAI embeddings
        
        Args:
            force_reload: Whether to force reloading even if index exists
            
        Returns:
            Success status
        """
        try:
            # Check if OpenAI API key is set
            if not os.environ.get("OPENAI_API_KEY") and not self.openai_api_key:
                logger.error("OpenAI API key not found. Please set OPENAI_API_KEY env var or provide it in constructor.")
                return False
            
            # Import LlamaIndex components
            from llama_index.core import (
                VectorStoreIndex, 
                SimpleDirectoryReader,
                StorageContext,
                load_index_from_storage,
                Settings
            )
            
            # Import OpenAI embeddings
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            # Check if we have an existing index
            if os.path.exists(INDEX_PERSIST_DIR) and not force_reload:
                logger.info(f"Loading existing index from {INDEX_PERSIST_DIR}")
                try:
                    # Configure settings with OpenAI embeddings but no LLM
                    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
                    Settings.embed_model = embed_model
                    Settings.chunk_size = 512
                    Settings.chunk_overlap = 50
                    Settings.llm = None  # Explicitly disable LLM component
                    
                    # Load existing index
                    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PERSIST_DIR)
                    self.index = load_index_from_storage(storage_context)
                    
                    # Create query engine
                    self.query_engine = self.index.as_query_engine()
                    
                    logger.info("Successfully loaded existing index")
                    return True
                except Exception as e:
                    logger.error(f"Error loading existing index: {e}")
                    logger.info("Will create a new index")
            
            # Create a new index
            
            # Check if therapy_documents directory exists
            if not os.path.exists(DOCS_PATH):
                logger.error(f"Therapy documents directory not found: {DOCS_PATH}")
                return False
            
            # Configure settings with OpenAI embeddings and no LLM
            embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            Settings.embed_model = embed_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            Settings.llm = None  # Explicitly disable LLM component
            
            logger.info(f"Using OpenAI embeddings for document indexing")
            
            # Load documents from the therapy_documents folder
            logger.info(f"Loading documents from {DOCS_PATH}")
            start_time = time.time()
            
            # Create document reader for all files in therapy_documents
            try:
                documents = SimpleDirectoryReader(
                    input_dir=DOCS_PATH,
                    recursive=True
                ).load_data()
            except Exception as e:
                logger.error(f"Error loading documents: {e}")
                return False
            
            if not documents:
                logger.warning("No documents were loaded. Check your therapy_documents folder")
                return False
            
            logger.info(f"Loaded {len(documents)} documents in {time.time() - start_time:.2f} seconds")
            
            # Create LlamaIndex service context for OpenAI
            embed_model = OpenAIEmbedding()
            service_context = ServiceContext.from_defaults(embed_model=embed_model)
            
            # Get or create Pinecone index for therapy knowledge
            try:
                # Get the Pinecone index
                self.pinecone_index_name = pinecone_manager.get_therapy_index()
                logger.info(f"Using Pinecone index: {self.pinecone_index_name}")
                
                # Import necessary classes for Pinecone with correct path
                try:
                    from llama_index.vector_stores.pinecone import PineconeVectorStore
                except ImportError:
                    # Try alternate import path
                    from llama_index.core.vector_stores.pinecone import PineconeVectorStore
                
                # Clear existing vectors in the index (for a fresh start)
                pinecone_manager.clear_index(self.pinecone_index_name)
                
                # Get Pinecone index
                pinecone_index = pinecone_manager.get_index(self.pinecone_index_name)
                
                # Create vector store with Pinecone
                vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
                
                # Create vector index with Pinecone as storage
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    service_context=service_context,
                    vector_store=vector_store,
                    show_progress=True
                )
                
                logger.info(f"Successfully stored vectors in Pinecone index: {self.pinecone_index_name}")
            except Exception as e:
                logger.error(f"Error using Pinecone, falling back to in-memory storage: {e}")
                # Fallback to in-memory storage
                self.index = VectorStoreIndex(documents, service_context=service_context)
                
            # Create directory for backup if needed
            os.makedirs(INDEX_PERSIST_DIR, exist_ok=True)
            
            # Set up optimized query engine with faster settings
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import VectorIndexRetriever
            
            # Create faster retriever with optimized settings
            retriever = VectorIndexRetriever(
                index=documents_to_index,
                similarity_top_k=3  # Limit results for faster retrieval
            )
            
            # Create optimized query engine with configurable parameters
            from llama_index.core.response_synthesizers import get_response_synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",  # More efficient response generation
                use_async=True  # Use async for better performance
            )
            
            # Create optimized query engine
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )
            
            logger.info(f"Successfully indexed {len(documents)} therapy documents with optimized retrieval")
            return True
            
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            logger.error("Make sure llama-index and openai are installed")
            return False
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False
    
    def query(self, query_text: str, similarity_top_k: int = 3, filter_by_relevance: bool = True) -> str:
        """
        Query the therapy knowledge base
        
        Args:
            query_text: The query to answer
            
        Returns:
            Relevant information from therapy documents
        """
        if not self.query_engine:
            if not self.load_documents():
                return "Error: The therapy knowledge base is not available."
        
        try:
            logger.info(f"Querying: {query_text}")
            start_time = time.time()
            
            # Check if query is in cache
            cache_key = query_text.strip().lower()
            if cache_key in self.query_cache:
                self.cache_hits += 1
                response = self.query_cache[cache_key]
                logger.info(f"Query answered from cache in {time.time() - start_time:.2f} seconds (hits: {self.cache_hits}, misses: {self.cache_misses})")
            else:
                # Execute query
                self.cache_misses += 1
                response = self.query_engine.query(query_text)
                
                # Update cache (simple LRU - remove oldest if full)
                if len(self.query_cache) >= self.cache_size:
                    # Remove oldest item
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                    
                # Add to cache
                self.query_cache[cache_key] = response
                
                logger.info(f"Query answered in {time.time() - start_time:.2f} seconds (hits: {self.cache_hits}, misses: {self.cache_misses})")
            
            # Format the response
            result = f"THERAPY KNOWLEDGE:\n\n{str(response)}"
            
            return result
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return f"Error querying therapy knowledge: {str(e)}"
    
    def get_therapy_knowledge(self, message: str) -> str:
        """
        Get relevant therapy knowledge based on a message
        
        This method is designed to be used with the Letta connector
        
        Args:
            message: The user message to extract knowledge for
            
        Returns:
            Relevant therapy knowledge
        """
        return self.query(message)


def main():
    """Test the OpenAI therapy indexing system"""
    logger.info("Testing OpenAI Therapy Knowledge Base with LlamaIndex")
    
    # Try to load API key from .env file in the project root
    env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    os.environ['OPENAI_API_KEY'] = line.split('=', 1)[1].strip().strip('"')
                    break
    
    # Check if API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found in environment variables or .env file.")
        print("Please check that your .env file contains a valid OPENAI_API_KEY.")
        return
    
    # Initialize the system with OpenAI embeddings
    therapy_index = OpenAITherapyIndex(openai_api_key=api_key)
    
    # Load and index all documents
    logger.info("Loading all documents from therapy_documents folder (forced reload)")
    success = therapy_index.load_documents(force_reload=True)
    
    if not success:
        logger.error("Failed to load and index documents")
        return
    
    # Test queries
    test_queries = [
        "How can I manage anxiety?",
        "What are some CBT techniques for depression?",
        "Tell me about panic attacks",
        "What are ways to practice mindfulness?",
        "How can I help clients with eating disorders?",
        "What are effective therapy interventions for trauma?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        start_time = time.time()
        result = therapy_index.query(query)
        print(f"Retrieved in {time.time() - start_time:.2f} seconds")
        
        print(result)
        print("=" * 60)


if __name__ == "__main__":
    main()
