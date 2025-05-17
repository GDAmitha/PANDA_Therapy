"""
Pinecone integration for PANDA Therapy

This module provides helper functions to integrate Pinecone vector database
with both therapy knowledge and transcript memory systems.
"""

import os
import time
import logging
import uuid
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
# Setup logging first to avoid referencing logger before assignment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Pinecone through our wrapper
import backend.pinecone_wrapper as pc

# Load environment variables

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

# Get API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Constants for index names
THERAPY_INDEX_NAME = "therapy-knowledge"
TRANSCRIPT_INDEX_PREFIX = "user-transcripts-"  # Will be followed by user_id

class PineconeManager:
    """
    Manager class for Pinecone integration that handles:
    1. Creation and management of indexes for therapy knowledge and user transcripts
    2. Helper functions for vector operations
    """
    
    def __init__(self):
        """Initialize the Pinecone manager"""
        if not PINECONE_API_KEY:
            logger.error("PINECONE_API_KEY not found in environment variables")
            raise ValueError("Pinecone API key is required")
        
        # Check if Pinecone wrapper is initialized
        if not pc:
            logger.error("Pinecone wrapper is not initialized")
            raise ValueError("Pinecone wrapper is not initialized")
            
        logger.info("Pinecone manager initialized successfully")
        self.indexes = {}
        
    def get_therapy_index(self, dimension: int = 1536) -> str:
        """
        Get or create the therapy knowledge index
        
        Args:
            dimension: Vector dimension (default: 1536 for OpenAI embeddings)
            
        Returns:
            Name of the index
        """
        # Check if index exists
        existing_indexes = pc.list_indexes()
        
        if THERAPY_INDEX_NAME not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {THERAPY_INDEX_NAME}")
            
            # Create index with pinecone-client API
            try:
                # Create index
                pc.create_index(
                    name=THERAPY_INDEX_NAME,
                    dimension=dimension,
                    metric="cosine"
                )
                
                # Wait for index to be ready
                import time
                ready = False
                retry_count = 0
                max_retries = 10
                while not ready and retry_count < max_retries:
                    try:
                        # Check if the index appears in the list
                        if THERAPY_INDEX_NAME in pc.list_indexes():
                            ready = True
                            logger.info(f"Index {THERAPY_INDEX_NAME} is ready")
                        else:
                            logger.info("Waiting for index to be ready...")
                            time.sleep(5)
                            retry_count += 1
                    except Exception as e:
                        logger.error(f"Error checking index readiness: {e}")
                        time.sleep(2)
                        retry_count += 1
            except Exception as e:
                logger.error(f"Failed to create Pinecone index: {e}")
                return ""
                
        return THERAPY_INDEX_NAME
    
    def get_transcript_index(self, user_id: str, dimension: int = 1536) -> str:
        """
        Get or create a transcript index for a specific user
        
        Args:
            user_id: User ID
            dimension: Vector dimension (default: 1536 for OpenAI embeddings)
            
        Returns:
            Name of the index
        """
        # Use sanitized index name (Pinecone has naming restrictions)
        # Convert user_id to lowercase alphanumeric only
        safe_user_id = ''.join(c for c in user_id if c.isalnum()).lower()
        index_name = f"{TRANSCRIPT_INDEX_PREFIX}{safe_user_id}"
        
        # Check if index exists
        existing_indexes = pc.list_indexes()
        
        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index for user {user_id}: {index_name}")
            
            # Create index with pinecone-client API
            try:
                # Create index
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine"
                )
                
                # Wait for index to be ready
                import time
                ready = False
                retry_count = 0
                max_retries = 10
                while not ready and retry_count < max_retries:
                    try:
                        # Check if the index appears in the list
                        if index_name in pc.list_indexes():
                            ready = True
                            logger.info(f"Index {index_name} is ready")
                        else:
                            logger.info("Waiting for index to be ready...")
                            time.sleep(5)
                            retry_count += 1
                    except Exception as e:
                        logger.error(f"Error checking index readiness: {e}")
                        time.sleep(2)
                        retry_count += 1
            except Exception as e:
                logger.error(f"Failed to create Pinecone index: {e}")
                return ""
                
        return index_name
    
    def delete_transcript_index(self, user_id: str) -> bool:
        """
        Delete a user's transcript index
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        # Use sanitized index name
        safe_user_id = ''.join(c for c in user_id if c.isalnum()).lower()
        index_name = f"{TRANSCRIPT_INDEX_PREFIX}{safe_user_id}"
        
        return pc.delete_index(index_name)
            
    def clear_index(self, index_name: str) -> bool:
        """
        Clear all vectors from an index
        
        Args:
            index_name: Name of the index to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the index
            index = pc.get_index(index_name)
            if not index:
                logger.error(f"Index not found: {index_name}")
                return False
                
            # Delete all vectors (using delete_all=True)
            index.delete(delete_all=True)
            
            logger.info(f"Cleared all vectors from index {index_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing index {index_name}: {e}")
            return False
    
    def list_indexes(self) -> List[str]:
        """List all available indexes"""
        return pc.list_indexes()
        
    def get_index(self, index_name: str):
        """Get an index by name"""
        return pc.get_index(index_name)

# Singleton instance
pinecone_manager = PineconeManager()
