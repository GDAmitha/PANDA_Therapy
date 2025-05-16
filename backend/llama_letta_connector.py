"""
LlamaIndex Connector for Letta Therapy Agents

This module integrates LlamaIndex RAG with Letta agents to enhance therapy responses
with knowledge from therapy documents.
"""

import os
import sys
import logging
import re
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the OpenAI-powered LlamaIndex implementation
try:
    from openai_llama_rag import OpenAITherapyIndex as TherapyKnowledgeBase
    logger.info("Using OpenAITherapyIndex with OpenAI embeddings for RAG")
except ImportError:
    logger.error("Could not import OpenAITherapyIndex from openai_llama_rag")
    TherapyKnowledgeBase = None


class LlamaLettaConnector:
    """
    Connector between LlamaIndex and Letta therapy agents
    
    This class enhances Letta agents with knowledge from therapy documents
    while maintaining separation between RAG and patient-specific memory.
    """
    
    def __init__(self):
        """Initialize the connector"""
        self.knowledge_base = TherapyKnowledgeBase() if TherapyKnowledgeBase else None
        self.initialized = False
        
        # Try to initialize if knowledge base is available
        if self.knowledge_base:
            try:
                success = self.knowledge_base.load_documents()
                self.initialized = success
                if success:
                    logger.info("LlamaIndex-Letta connector initialized successfully")
                else:
                    logger.error("Failed to initialize LlamaIndex-Letta connector")
            except Exception as e:
                logger.error(f"Error initializing LlamaIndex-Letta connector: {e}")
    
    def extract_therapy_questions(self, message: str) -> List[str]:
        """
        Extract potential therapy questions from a message
        
        Args:
            message: User message to analyze
            
        Returns:
            List of potential therapy questions
        """
        # Extract questions marked with ? or implied questions
        questions = []
        
        # Direct questions
        question_marks = message.split('?')
        for i in range(len(question_marks) - 1):
            # Find the start of the question (after last period or beginning of text)
            start = question_marks[i].rfind('.')
            if start == -1:
                start = 0
            else:
                start += 1
                
            question = question_marks[i][start:].strip() + '?'
            if len(question) > 10:  # Avoid very short questions
                questions.append(question)
        
        # Implied questions/topics (if no direct questions found)
        if not questions:
            # Look for therapy-related keywords
            therapy_keywords = [
                'anxious', 'anxiety', 'depression', 'stressed', 'therapy', 
                'trauma', 'panic', 'worry', 'mental health', 'coping',
                'mindfulness', 'meditation', 'cbt', 'dbt', 'technique'
            ]
            
            for keyword in therapy_keywords:
                if keyword.lower() in message.lower():
                    # Extract the sentence containing the keyword
                    sentences = re.split(r'[.!?]+', message)
                    for sentence in sentences:
                        if keyword.lower() in sentence.lower():
                            clean_sentence = sentence.strip()
                            if clean_sentence:
                                questions.append(clean_sentence)
        
        # If still no questions, use the whole message as a general query
        if not questions and len(message) > 0:
            questions.append(message)
            
        return questions
    
    def get_therapy_knowledge(self, message: str, max_tokens: int = 800) -> str:
        """
        Get relevant therapy knowledge for a user message
        
        Args:
            message: The user message to find relevant knowledge for
            max_tokens: Maximum number of tokens to return (approximate)
            
        Returns:
            Relevant therapy knowledge or empty string if not available
        """
        if not self.initialized or not self.knowledge_base:
            logger.warning("LlamaIndex RAG is not initialized")
            return ""
        
        try:
            # Extract potential therapy questions
            questions = self.extract_therapy_questions(message)
            
            if not questions:
                logger.info("No therapy questions found in message")
                return ""
                
            # Query the knowledge base for each question and combine results
            combined_results = ""
            for question in questions:
                logger.info(f"Querying knowledge base with: {question}")
                result = self.knowledge_base.query(question)
                
                if result and len(result) > 20:  # Check if we got meaningful results
                    if combined_results:
                        combined_results += "\n\n---\n\n"
                    combined_results += result
            
            # Trim to fit token limit (approximate)
            if combined_results and len(combined_results) > max_tokens * 4:  # Rough approximation: 4 chars ≈ 1 token
                combined_results = combined_results[:max_tokens * 4] + "..."
                
            return combined_results
        except Exception as e:
            logger.error(f"Error getting therapy knowledge: {e}")
            return ""
    
    def enhance_prompt(self, message: str, system_prompt: str) -> str:
        """
        Enhance a Letta system prompt with relevant therapy knowledge
        
        Args:
            message: The user message
            system_prompt: The original system prompt
            
        Returns:
            Enhanced system prompt with relevant therapy knowledge
        """
        therapy_knowledge = self.get_therapy_knowledge(message)
        
        if not therapy_knowledge:
            return system_prompt
        
        # Add the therapy knowledge to the system prompt
        enhanced_prompt = system_prompt + "\n\n"
        enhanced_prompt += "RELEVANT THERAPY KNOWLEDGE:\n" + therapy_knowledge
        
        return enhanced_prompt
    
    def get_letta_response(self, client, agent_id, message: str, system_prompt: str = None):
        """
        Get a response from a Letta agent with enhanced therapy knowledge
        
        Args:
            client: Letta client instance
            agent_id: ID of the Letta agent
            message: User message
            system_prompt: Optional system prompt to use
            
        Returns:
            Agent response with enhanced therapy knowledge
        """
        try:
            # If we have a system prompt and RAG is initialized, enhance it
            if system_prompt and self.initialized:
                enhanced_prompt = self.enhance_prompt(message, system_prompt)
                
                # Use enhanced prompt with Letta
                response = client.agents.messages.create(
                    agent_id=agent_id,
                    messages=[{"role": "user", "content": message}],
                    system=enhanced_prompt
                )
            else:
                # Use default Letta behavior
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
            return f"I'm having trouble connecting to the therapy knowledge. Error: {e}"


# Test function
def main():
    """Test the LlamaLetta connector"""
    connector = LlamaLettaConnector()
    
    if not connector.initialized:
        logger.error("Failed to initialize connector")
        return
    
    # Test with some example messages
    test_messages = [
        "I've been feeling anxious lately, what can I do?",
        "How can I help a client with depression?",
        "What techniques work for panic attacks?",
        "I need some mindfulness exercises for my patients",
        "I'm worried all the time and can't sleep. Any advice?",
        "My therapist suggested CBT but I don't know what that is."
    ]
    
    original_prompt = "You are a helpful therapeutic assistant. Provide compassionate and evidence-based guidance."
    
    for message in test_messages:
        print(f"\nUser Message: {message}")
        print("-" * 40)
        
        # Get therapy knowledge
        therapy_knowledge = connector.get_therapy_knowledge(message)
        
        # Print the extracted knowledge (truncated for readability)
        max_display = 300
        if therapy_knowledge:
            if len(therapy_knowledge) > max_display:
                display_knowledge = therapy_knowledge[:max_display] + "... [truncated]"
            else:
                display_knowledge = therapy_knowledge
                
            print(f"Retrieved therapy knowledge: {display_knowledge}")
        else:
            print("No relevant therapy knowledge found")
            
        print("=" * 60)


if __name__ == "__main__":
    main()
