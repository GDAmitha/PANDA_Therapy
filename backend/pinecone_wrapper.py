"""
Pinecone wrapper to handle different versions of Pinecone packages.

This module provides a unified interface for the Pinecone package regardless
of which version is installed (pinecone-client or pinecone).
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Direct approach - set the Pinecone API key directly
# This is a workaround since the key isn't being loaded from .env properly
PINECONE_API_KEY = "pcsk_4rhGRe_6oer72FpJ8uijorczTJEfRs9UDNLjZ1KU5yHAYKmtWcrYAmH7pk7jusyzWsfmJy"

# Set it in the environment for other components to use
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

logger.info("PINECONE_API_KEY set directly in code")

# Global wrapper instance
pc = None

try:
    # Try newer Pinecone package first (direct Pinecone class)
    try:
        from pinecone import Pinecone, ServerlessSpec
        
        # Create a wrapper class for compatibility with the rest of the codebase
        class PineconeWrapper:
            """Wrapper class for newer Pinecone client (v2+)"""
            
            def __init__(self, api_key):
                """Initialize with Pinecone client"""
                self.pc = Pinecone(api_key=api_key)
                
            def is_ready(self) -> bool:
                """Check if Pinecone is ready"""
                return self.pc is not None
                
            def list_indexes(self):
                """List available indexes"""
                return self.pc.list_indexes().names()
            
            def create_index(self, name: str, dimension: int, metric: str = "cosine", **kwargs) -> bool:
                """Create a new index"""
                try:
                    # Create with serverless spec by default
                    spec = kwargs.get('spec', ServerlessSpec(cloud="aws", region="us-west-2"))
                    self.pc.create_index(
                        name=name,
                        dimension=dimension,
                        metric=metric,
                        spec=spec
                    )
                    time.sleep(1)
                    return True
                except Exception as e:
                    logger.error(f"Error creating index: {e}")
                    return False
                    
            def delete_index(self, name: str) -> bool:
                """Delete an index"""
                try:
                    self.pc.delete_index(name)
                    return True
                except Exception as e:
                    logger.error(f"Error deleting index: {e}")
                    return False
                    
            def describe_index(self, name: str) -> Dict[str, Any]:
                """Get details about an index"""
                try:
                    # New API doesn't have describe_index, so simulate it
                    index_list = self.pc.list_indexes().names()
                    if name in index_list:
                        return {"status": {"ready": True}}
                    return {}
                except Exception as e:
                    logger.error(f"Error describing index: {e}")
                    return {}
                    
            def get_index(self, name: str):
                """Get a Pinecone index"""
                try:
                    return self.pc.Index(name)
                except Exception as e:
                    logger.error(f"Error getting index: {e}")
                    return None
        
        # Initialize with newer Pinecone client
        pc = PineconeWrapper(PINECONE_API_KEY)
        logger.info("Pinecone initialized successfully with new Pinecone API")
        
    except (ImportError, AttributeError) as e:
        # Fall back to older pinecone-client (v1.x)
        import pinecone
        
        # Create a wrapper class for compatibility with the rest of the codebase
        class LegacyPineconeWrapper:
            """Wrapper class for older pinecone-client package"""
            
            def __init__(self, api_key):
                """Initialize with legacy Pinecone client"""
                # Check for different initialization methods
                if hasattr(pinecone, 'init'):
                    # Old style init
                    pinecone.init(api_key=api_key)
                    self.pc = pinecone
                else:
                    # Try direct Pinecone constructor if available
                    self.pc = pinecone.Pinecone(api_key=api_key)
                
            def is_ready(self) -> bool:
                """Check if Pinecone is ready"""
                return self.pc is not None
                
            def list_indexes(self):
                """List available indexes"""
                if hasattr(self.pc, 'list_indexes'):
                    return self.pc.list_indexes()
                return []
            
            def create_index(self, name: str, dimension: int, metric: str = "cosine", **kwargs) -> bool:
                """Create a new index"""
                try:
                    if hasattr(self.pc, 'create_index'):
                        self.pc.create_index(
                            name=name,
                            dimension=dimension,
                            metric=metric
                        )
                    else:
                        logger.error("create_index method not found")
                        return False
                    time.sleep(1)
                    return True
                except Exception as e:
                    logger.error(f"Error creating index: {e}")
                    return False
                    
            def delete_index(self, name: str) -> bool:
                """Delete an index"""
                try:
                    if hasattr(self.pc, 'delete_index'):
                        self.pc.delete_index(name)
                    else:
                        logger.error("delete_index method not found")
                        return False
                    return True
                except Exception as e:
                    logger.error(f"Error deleting index: {e}")
                    return False
                    
            def describe_index(self, name: str) -> Dict[str, Any]:
                """Get details about an index"""
                try:
                    if hasattr(self.pc, 'describe_index'):
                        return self.pc.describe_index(name)
                    return {}
                except Exception as e:
                    logger.error(f"Error describing index: {e}")
                    return {}
                    
            def get_index(self, name: str):
                """Get a Pinecone index"""
                try:
                    if hasattr(self.pc, 'Index'):
                        return self.pc.Index(name)
                    logger.error("Index method not found")
                    return None
                except Exception as e:
                    logger.error(f"Error getting index: {e}")
                    return None
        
        # Initialize with legacy pinecone-client
        pc = LegacyPineconeWrapper(PINECONE_API_KEY)
        logger.info("Pinecone initialized successfully with legacy pinecone-client")
    
except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    pc = None

# Module functions that delegate to the wrapper instance
def list_indexes() -> List[str]:
    """List all available Pinecone indexes"""
    if pc:
        try:
            return pc.list_indexes()
        except Exception as e:
            logger.error(f"Error listing indexes: {e}")
    return []

def create_index(name: str, dimension: int, metric: str = "cosine", **kwargs) -> bool:
    """Create a Pinecone index"""
    if pc:
        return pc.create_index(name, dimension, metric, **kwargs)
    return False

def delete_index(name: str) -> bool:
    """Delete a Pinecone index"""
    if pc:
        return pc.delete_index(name)
    return False

def describe_index(name: str) -> Dict[str, Any]:
    """Get details about a Pinecone index"""
    if pc:
        return pc.describe_index(name)
    return {}

def get_index(name: str):
    """Get a Pinecone index"""
    if pc:
        return pc.get_index(name)
    return None

def is_ready() -> bool:
    """Check if Pinecone is available and initialized"""
    return pc is not None and pc.is_ready()
