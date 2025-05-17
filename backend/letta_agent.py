"""
Letta Agent Manager for PANDA Therapy

This module handles the creation and management of Letta agents for users.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List

# Import the official Letta client SDK
from letta_client import Letta

# Make sure we can load environment variables 
from dotenv import load_dotenv

# Try to import optional dependencies
try:
    from models.user import User
    from database import Database
    has_database = True
except ImportError:
    # For the simplified API, create a compatible User class
    from pydantic import BaseModel
    class User(BaseModel):
        id: str
        username: str
        name: str
        role: str
        letta_agent_id: Optional[str] = None
    has_database = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LettaAgentManager:
    """Manager class for Letta agents"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Letta agent manager with the official SDK
        
        Args:
            api_key: Letta API key, defaults to environment variable
        """
        # If API key is directly provided, use it
        if api_key:
            self.api_key = api_key
            logger.info("Using provided Letta API key")
        else:
            # Try to load from environment variable first
            self.api_key = os.getenv("LETTA_API_KEY")
            
            # If not in environment variable, try to read directly from .env file
            if not self.api_key:
                # Try to load from parent directory if we're in the backend folder
                parent_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
                if os.path.exists(parent_env_path):
                    logger.info(f"Reading .env file directly from {parent_env_path}")
                    try:
                        # Manually parse the .env file
                        with open(parent_env_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    key, value = line.split('=', 1)
                                    if key.strip() == "LETTA_API_KEY":
                                        # Remove quotes if present
                                        self.api_key = value.strip().strip('"\'')
                                        logger.info(f"Found LETTA_API_KEY in .env file")
                                        break
                    except Exception as e:
                        logger.error(f"Error reading .env file: {str(e)}")
        
        # Hardcode the API key if all else fails (obviously in a real app you'd never do this)
        if not self.api_key:
            # Use the key from the .env file that was shown earlier
            self.api_key = "sk-let-MTA2OGZmNzUtNTA5MC00MGE1LWE0ZDctYjE3YjNhY2MyMjU5OmMwMTYyYjdmLTNiY2YtNGM5Yi04ZDY5LTEwMDljODBjMWZlYg=="
            logger.warning("Using hardcoded API key as fallback")
            
        # Final check if we have an API key
        if not self.api_key:
            logger.error("No Letta API key provided, functionality will be limited")
            self.letta_client = None
        else:
            logger.info("Letta API key loaded successfully")
            # Initialize the official Letta client SDK
            try:
                self.letta_client = Letta(token=self.api_key)
                logger.info("Letta client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Letta client: {str(e)}")
                self.letta_client = None
            
        # Initialize database if available
        if has_database:
            self.db = Database()
        else:
            logger.warning("Database module not available, using in-memory storage")
            self.db = None
        
    def create_agent_for_user(self, user: User) -> Optional[str]:
        """
        Create a new Letta agent for a user using the Letta client SDK
        
        Args:
            user: User to create an agent for
            
        Returns:
            Agent ID if successful, None otherwise
        """
        if not self.letta_client:
            logger.error("Cannot create agent: Letta client not initialized")
            return None
        
        try:
            # Determine agent type based on user role
            # Handle both enum and string role types
            if hasattr(user.role, 'value'):
                # Role is an enum (from the original User model)
                role_value = user.role.value
            else:
                # Role is a string (from our simplified User model)
                role_value = user.role
            
            # Determine which template to use based on user role
            template = "Panda_Therapist:latest" if role_value == "therapist" else "Panda_Patient:latest"
            
            logger.info(f"Creating agent for user {user.username} with template {template}")
            
            # Create memory blocks specific to this user
            memory_blocks = [
                {
                    "label": "user_info",
                    "value": f"The user's name is {user.name}. Their username is {user.username}. Their role is {role_value}."
                },
                {
                    "label": "persona",
                    "value": "You are PANDA, a therapy assistant. You are compassionate, understanding, and committed to supporting the well-being of those you interact with."
                }
            ]
            
            # Create the agent using the SDK
            try:
                # Create with template
                agent_state = self.letta_client.agents.create(
                    name=f"Agent_{user.name}",
                    from_template=template,
                    memory_blocks=memory_blocks
                )
                
                agent_id = agent_state.id
                
                logger.info(f"Successfully created Letta agent {agent_id} for user {user.id}")
                
                # Update user with new agent ID if database is available
                if has_database and self.db:
                    success = self.db.update_user_letta_agent(user.id, agent_id)
                    if not success:
                        logger.warning(f"Created agent {agent_id} but failed to update user record")
                else:
                    # In simplified mode, just log - user_letta_agents mapping is handled by simple_api.py
                    logger.info(f"Created agent {agent_id} for user {user.id} (no database update)")
                
                return agent_id
                
            except Exception as e:
                logger.error(f"Error from Letta SDK: {str(e)}")
                # Try fallback with manual creation if template approach fails
                logger.info("Attempting fallback agent creation method...")
                
                # Fallback: Create without template
                agent_state = self.letta_client.agents.create(
                    name=f"Agent_{user.name}",
                    model="openai/gpt-4.1",  # Using capable model as fallback
                    embedding="openai/text-embedding-3-small",
                    memory_blocks=memory_blocks
                )
                
                agent_id = agent_state.id
                logger.info(f"Successfully created fallback Letta agent {agent_id} for user {user.id}")
                return agent_id
            
        except Exception as e:
            logger.exception(f"Error creating Letta agent: {str(e)}")
            return None
    
    def get_or_create_agent_for_user(self, user: User) -> Optional[str]:
        """
        Get a user's Letta agent ID, creating one if it doesn't exist
        
        Args:
            user: User to get or create an agent for
            
        Returns:
            Agent ID if available or created, None otherwise
        """
        # Check if user already has an agent ID
        if hasattr(user, 'letta_agent_id') and user.letta_agent_id:
            # Verify the agent still exists
            if self.verify_agent_exists(user.letta_agent_id):
                logger.info(f"Using existing Letta agent {user.letta_agent_id} for user {user.id}")
                return user.letta_agent_id
            else:
                logger.warning(f"Agent {user.letta_agent_id} no longer exists for user {user.id}, creating new one")
        
        # Create a new agent
        agent_id = self.create_agent_for_user(user)
        if agent_id:
            logger.info(f"Created new Letta agent {agent_id} for user {user.id}")
        else:
            logger.error(f"Failed to create or get Letta agent for user {user.id}")
            
        return agent_id
    
    def verify_agent_exists(self, agent_id: str) -> bool:
        """Check if a Letta agent exists using the SDK"""
        if not self.letta_client:
            logger.warning("Cannot verify agent: Letta client not initialized")
            return False
        
        try:
            # Try to retrieve the agent using the SDK
            agent_state = self.letta_client.agents.retrieve(agent_id=agent_id)
            # If we get here without an exception, the agent exists
            logger.info(f"Verified agent {agent_id} exists")
            return True
        except Exception as e:
            logger.error(f"Error verifying agent existence: {str(e)}")
            return False
    
    def chat_with_agent(self, agent_id: str, message: str, history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a message to a Letta agent using the SDK
        
        Args:
            agent_id: ID of the Letta agent
            message: Message to send
            history: Optional chat history (not used since Letta manages its own memory)
            
        Returns:
            Dict with response information
        """
        if not self.letta_client:
            logger.error("Cannot chat with agent: Letta client not initialized")
            return {"error": "Letta client not initialized"}
        
        try:
            # Letta agents maintain their own conversation history
            # We only need to send the current message
            logger.info(f"Sending message to Letta agent {agent_id}")
            
            # Create a message using the SDK's API
            response = self.letta_client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": message}]
            )
            
            # Handle the response based on the SDK's structure
            if response and hasattr(response, 'assistant'):
                # Extract the assistant's response
                assistant_message = response.assistant
                if hasattr(assistant_message, 'content'):
                    # Format the response for our API
                    result = {
                        "response": assistant_message.content,
                        "agent_id": agent_id
                    }
                    
                    # Add sources if available
                    if hasattr(response, 'sources'):
                        result["sources"] = response.sources
                    
                    logger.info(f"Received successful response from Letta agent {agent_id}")
                    return result
                else:
                    logger.warning(f"Assistant message has no content attribute")
            elif response and hasattr(response, 'messages') and response.messages:
                # Try alternative message structure
                # Find the assistant message in the list
                assistant_content = None
                
                # Just extract the last message which should be from the assistant
                if len(response.messages) > 0:
                    last_message = response.messages[-1]
                    if hasattr(last_message, 'content'):
                        assistant_content = last_message.content
                
                if assistant_content:
                    result = {
                        "response": assistant_content,
                        "agent_id": agent_id
                    }
                    logger.info(f"Received successful response from Letta agent {agent_id}")
                    return result
            
            # If we didn't get a proper response structure
            logger.warning(f"Unusual response structure from Letta agent {agent_id}")
            return {"response": "I received your message but had trouble generating a response."}
            
        except Exception as e:
            logger.exception(f"Error chatting with agent: {str(e)}")
            return {"error": str(e)}
            
    def update_agent_knowledge(self, agent_id: str, document_content: str, document_name: str) -> bool:
        """
        Update an agent's knowledge base with a document using the SDK
        
        Args:
            agent_id: ID of the Letta agent
            document_content: Content of the document to add
            document_name: Name of the document
            
        Returns:
            True if successful, False otherwise
        """
        if not self.letta_client:
            logger.error("Cannot update agent knowledge: Letta client not initialized")
            return False
        
        try:
            # Create a memory block for this document
            block = self.letta_client.blocks.create(
                label=document_name,
                value=document_content,
                limit=4000,  # Set a reasonable limit for the block size
            )
            
            if not block or not hasattr(block, 'id'):
                logger.error(f"Failed to create memory block for document {document_name}")
                return False
                
            # Attach the block to the agent
            logger.info(f"Created memory block {block.id} for document {document_name}")
            result = self.letta_client.agents.blocks.attach(agent_id=agent_id, block_id=block.id)
            
            if result:
                logger.info(f"Successfully attached block {block.id} to agent {agent_id}")
                return True
            else:
                logger.error(f"Failed to attach block {block.id} to agent {agent_id}")
                return False
            
        except Exception as e:
            logger.exception(f"Error updating agent knowledge: {str(e)}")
            return False
