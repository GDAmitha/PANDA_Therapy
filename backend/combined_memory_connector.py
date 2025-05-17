"""
Combined Memory Connector for Letta Therapy Agents

This module integrates both therapy knowledge (from documents) and 
patient-specific transcript memory for enhanced Letta responses.
"""

import os
import sys
import logging
import time
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the compatibility layer for RAG components
from backend.rag_compatibility import get_therapy_index, get_transcript_memory, full_rag_available

# Set RAG availability flag
rag_available = True  # Always true since we have fallbacks


class CombinedMemoryConnector:
    """
    Connector that combines therapy knowledge and patient-specific transcript memory
    
    This class enhances Letta prompts with both:
    1. General therapy knowledge from documents (using LlamaIndex)
    2. Patient-specific memories from past therapy sessions (using TherapyTranscriptMemory)
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the combined memory connector
        
        Args:
            user_id: User ID for transcript memory retrieval
        """
        self.user_id = user_id
        self.therapy_knowledge = None
        self.transcript_memory = None
        self.knowledge_initialized = False
        self.memory_initialized = False
        
        # Initialize therapy knowledge (RAG from documents)
        try:
            # Create and initialize therapy knowledge using the factory function
            self.therapy_knowledge = get_therapy_index()
            success = self.therapy_knowledge.load_documents()
            self.knowledge_initialized = success
            
            if success:
                logger.info("Therapy knowledge initialized successfully")
                logger.info(f"Using {'full RAG' if full_rag_available else 'fallback'} mode for therapy knowledge")
            else:
                logger.warning("Failed to initialize therapy knowledge")
            
            # No return in __init__
        except Exception as e:
            logger.error(f"Error initializing therapy knowledge: {e}")
            # No return in __init__
    
    def initialize_user_memory(self, user_id: str) -> bool:
        """Initialize the transcript memory component for a user"""
        if self.memory_initialized:
            return True
        
        # Special case for user with UUID d5679ce8-c1b3-42d8-a289-2c8105905216
        # Map to 'natey' to use existing transcripts
        memory_user_id = user_id
        if user_id == "d5679ce8-c1b3-42d8-a289-2c8105905216":
            memory_user_id = "natey"
            logger.info(f"Using 'natey' as user ID for transcript memory instead of {user_id}")
            
        try:
            # Create and initialize transcript memory using the factory function
            self.transcript_memory = get_transcript_memory(memory_user_id)
            success = self.transcript_memory.create_memory_index()
            self.memory_initialized = success
            
            if success:
                logger.info(f"Transcript memory initialized for user {memory_user_id}")
                logger.info(f"Using {'full RAG' if full_rag_available else 'fallback'} mode for transcript memory")
            else:
                logger.warning(f"Failed to initialize transcript memory for user {memory_user_id}")
        except Exception as e:
            logger.error(f"Error initializing transcript memory: {e}")
    
    def get_therapy_knowledge(self, message: str) -> str:
        """Get relevant therapy knowledge for a message"""
        if not self.knowledge_initialized or not self.therapy_knowledge:
            logger.warning(f"Therapy knowledge not initialized for user {self.user_id}")
            return ""
        
        try:
            logger.info(f"Retrieving therapy knowledge for query: '{message[:50]}...' (user: {self.user_id})")
            start_time = time.time()
            result = self.therapy_knowledge.query(message)
            elapsed = time.time() - start_time
            
            # Truncate the result for logging to avoid overwhelming logs
            log_result = result[:100] + "..." if len(result) > 100 else result
            logger.info(f"Retrieved therapy knowledge in {elapsed:.2f}s: '{log_result}'")
            return result
        except Exception as e:
            logger.error(f"Error getting therapy knowledge: {e}")
            return ""
    
    def get_transcript_memory(self, message: str) -> str:
        """Get relevant transcript memories for a message"""
        if not self.memory_initialized or not self.transcript_memory:
            logger.warning(f"Transcript memory not initialized for user {self.user_id}")
            return ""
        
        try:
            logger.info(f"Retrieving transcript memory for query: '{message[:50]}...' (user: {self.user_id})")
            start_time = time.time()
            result = self.transcript_memory.extract_memories_for_letta(message)
            elapsed = time.time() - start_time
            
            # Truncate the result for logging to avoid overwhelming logs
            log_result = result[:100] + "..." if len(result) > 100 else result
            logger.info(f"Retrieved transcript memory in {elapsed:.2f}s: '{log_result}'")
            return result
        except Exception as e:
            logger.error(f"Error getting transcript memory: {e}")
            return ""
            
    def reload_therapy_knowledge(self) -> bool:
        """Reload the therapy knowledge index"""
        try:
            # Reinitialize the therapy knowledge
            self.therapy_knowledge = get_therapy_index()
            success = self.therapy_knowledge.load_documents()
            self.knowledge_initialized = success
            
            if success:
                logger.info("Therapy knowledge reloaded successfully")
            else:
                logger.error("Failed to reload therapy knowledge")
                
            return success
        except Exception as e:
            logger.error(f"Error reloading therapy knowledge: {e}")
            return False
    
    def reload_transcript_memory(self) -> bool:
        """Reload the transcript memory for the user"""
        try:
            # Reinitialize the transcript memory using the factory function
            self.transcript_memory = get_transcript_memory(self.user_id)
            success = self.transcript_memory.create_memory_index(force_reload=True)
            self.memory_initialized = success
            
            if success:
                logger.info(f"Transcript memory reloaded successfully for user {self.user_id}")
                logger.info(f"Using {'full RAG' if full_rag_available else 'fallback'} mode for transcript memory")
            else:
                logger.error(f"Failed to reload transcript memory for user {self.user_id}")
                
            return success
        except Exception as e:
            logger.error(f"Error reloading transcript memory: {e}")
            return False
    
    def enhance_prompt(self, message: str, system_prompt: str) -> str:
        """
        Enhance a Letta system prompt with both therapy knowledge and transcript memory
        
        Args:
            message: The user message
            system_prompt: The original system prompt
            
        Returns:
            Enhanced system prompt
        """
        # Start with the original prompt
        enhanced_prompt = system_prompt
        
        # Add therapy knowledge if available
        therapy_knowledge = self.get_therapy_knowledge(message)
        if therapy_knowledge:
            enhanced_prompt += "\n\n" + therapy_knowledge
        
        # Add transcript memory if available
        transcript_memory = self.get_transcript_memory(message)
        if transcript_memory:
            enhanced_prompt += "\n\n" + transcript_memory
        
        return enhanced_prompt
    
    def get_letta_response(self, client, agent_id: str, message: str, system_prompt: str = None):
        """
        Get a response from Letta with enhanced therapy knowledge and transcript memory
        
        Args:
            client: Letta client instance
            agent_id: ID of the Letta agent
            message: User message
            system_prompt: Optional system prompt to use
            
        Returns:
            Enhanced agent response
        """
        try:
            # If we have a system prompt, enhance it
            if system_prompt:
                enhanced_prompt = self.enhance_prompt(message, system_prompt)
                
                # Use enhanced prompt with Letta
                logger.info("Sending enhanced prompt to Letta")
                response = client.agents.messages.create(
                    agent_id=agent_id,
                    messages=[{"role": "user", "content": message}],
                    system=enhanced_prompt
                )
            else:
                # Use default Letta behavior
                logger.info("Using default Letta behavior (no prompt enhancement)")
                response = client.agents.messages.create(
                    agent_id=agent_id,
                    messages=[{"role": "user", "content": message}]
                )
                
            # Extract assistant response
            if response and hasattr(response, 'assistant') and hasattr(response.assistant, 'content'):
                return response.assistant.content
            else:
                return "I'm not sure how to respond to that."
                
        except Exception as e:
            logger.error(f"Error getting Letta response: {e}")
            return f"I'm having trouble accessing the therapy system. Error: {e}"


def main():
    """Test the combined memory connector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Combined Memory Connector')
    parser.add_argument('--user_id', type=str, required=True, help='User ID to retrieve memories for')
    parser.add_argument('--message', type=str, required=True, help='User message to test')
    args = parser.parse_args()
    
    # Initialize the combined memory connector
    connector = CombinedMemoryConnector(args.user_id)
    
    # Test system prompt enhancement
    test_prompt = "You are a helpful therapeutic assistant. Provide compassionate and evidence-based guidance."
    
    # Enhance prompt with therapy knowledge and transcript memory
    enhanced_prompt = connector.enhance_prompt(args.message, test_prompt)
    
    # Print the results
    print("\nORIGINAL PROMPT:")
    print("-" * 40)
    print(test_prompt)
    
    print("\nENHANCED PROMPT:")
    print("-" * 40)
    
    # Limit displayed length for readability
    max_display = 2000
    if len(enhanced_prompt) > max_display:
        print(enhanced_prompt[:max_display] + "... [truncated]")
    else:
        print(enhanced_prompt)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # If run without arguments, print usage
        print("Usage: python combined_memory_connector.py --user_id USER_ID --message \"User message\"")
        print("\nExample:")
        print("  python combined_memory_connector.py --user_id Dr.Smith --message \"I'm feeling anxious again like we discussed last time\"")
