"""
Transcript Memory System using OpenAI Embeddings

This module implements a memory system for therapy transcripts using OpenAI embeddings
to enable Letta to recall specific details from previous therapy sessions.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import re

# Import Pinecone manager
from pinecone_store import pinecone_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path setup
BASE_PATH = Path(__file__).parent.parent.absolute()
TRANSCRIPTS_PATH = os.path.join(BASE_PATH, "backend", "database", "transcripts.json")
USER_MEMORY_DIR = os.path.join(BASE_PATH, "backend", "user_memories")

class TherapyTranscriptMemory:
    """
    Memory system for therapy transcripts using OpenAI embeddings
    
    This class provides tools to:
    1. Load and process transcript data for a specific user
    2. Create embeddings of therapy conversations
    3. Query these embeddings to recall relevant past conversations
    4. Integrate with Letta for enhanced patient-specific responses
    """
    
    def __init__(self, user_id: str, openai_api_key: Optional[str] = None, cache_size: int = 30):
        """
        Initialize the transcript memory system
        
        Args:
            user_id: Unique identifier for the user/patient
            openai_api_key: OpenAI API key (optional, will use env var if not provided)
            cache_size: Size of the query cache (default: 30 entries)
        """
        self.user_id = user_id
        self.openai_api_key = openai_api_key
        self.index = None
        self.query_engine = None
        self.transcripts = []
        self.memory_dir = os.path.join(USER_MEMORY_DIR, f"user_{user_id.lower()}")
        
        # Simple cache for transcript memory queries
        self.query_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Set OpenAI API key if provided
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            
        # Initialize Pinecone index name for this user
        self.pinecone_index_name = None
        
        # Create user memory directory if it doesn't exist
        os.makedirs(self.memory_dir, exist_ok=True)
    
    def load_transcripts(self) -> bool:
        """
        Load transcripts for the specified user from the transcripts.json file
        
        Returns:
            Success status
        """
        try:
            if not os.path.exists(TRANSCRIPTS_PATH):
                logger.error(f"Transcripts file not found: {TRANSCRIPTS_PATH}")
                return False
            
            # Load all transcripts
            with open(TRANSCRIPTS_PATH, 'r') as f:
                data = json.load(f)
            
            all_transcripts = []
            for session_id, transcript_data in data.items():
                # Add session_id to transcript data if not already present
                if 'session_id' not in transcript_data:
                    transcript_data['session_id'] = session_id
                    
                all_transcripts.append(transcript_data)
            
            # Try different methods to find relevant transcripts for this user
            self.transcripts = []
            
            # Method 1: Look for structured client data (newer format)
            for transcript in all_transcripts:
                # Check for structured client info
                if 'client' in transcript and isinstance(transcript['client'], dict):
                    client = transcript['client']
                    if client.get('id') == self.user_id or client.get('name') == self.user_id:
                        self.transcripts.append(transcript)
                        continue
                        
                # Method 2: Check for user_id in the transcript data (legacy format)
                if transcript.get('user_id') == self.user_id:
                    self.transcripts.append(transcript)
                    continue
            
            # If still no transcripts found, look by name in the speakers
            if not self.transcripts and not self.user_id.isdigit() and len(self.user_id) > 2:
                for transcript in all_transcripts:
                    # Try to find the name in transcript speakers
                    found_in_speakers = False
                    
                    # Try different transcript formats
                    if 'transcript' in transcript and isinstance(transcript['transcript'], list):
                        for entry in transcript['transcript']:
                            if self.user_id.lower() in entry.get('speaker', '').lower():
                                self.transcripts.append(transcript)
                                found_in_speakers = True
                                break
                    
                    if found_in_speakers:
                        continue
                        
                    # Also check other fields that might contain the name
                    transcript_str = json.dumps(transcript).lower()
                    if self.user_id.lower() in transcript_str:
                        if transcript not in self.transcripts:
                            self.transcripts.append(transcript)
            
            logger.info(f"Loaded {len(self.transcripts)} transcripts for user {self.user_id}")
            return len(self.transcripts) > 0
            
        except Exception as e:
            logger.error(f"Error loading transcripts: {e}")
            return False
    
    def create_memory_index(self, force_reload: bool = False) -> bool:
        """
        Create a searchable index of transcript content using OpenAI embeddings
        
        Args:
            force_reload: Whether to force reloading even if index exists
            
        Returns:
            Success status
        """
        try:
            # Check if OpenAI API key is set
            if not os.environ.get("OPENAI_API_KEY"):
                logger.error("OpenAI API key not found. Please set OPENAI_API_KEY env var.")
                return False
            
            # Import LlamaIndex components
            from llama_index.core import (
                VectorStoreIndex,
                Document,
                StorageContext,
                load_index_from_storage,
                Settings
            )
            
            # Import OpenAI embeddings
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            # Check if we have existing memory for this user
            memory_file = os.path.join(self.memory_dir, "memory_index")
            if os.path.exists(memory_file) and not force_reload:
                logger.info(f"Loading existing memory index for user {self.user_id}")
                try:
                    # Configure settings with OpenAI embeddings but no LLM
                    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
                    Settings.embed_model = embed_model
                    Settings.chunk_size = 512
                    Settings.chunk_overlap = 50
                    Settings.llm = None  # Explicitly disable LLM component
                    
                    # Load existing index
                    storage_context = StorageContext.from_defaults(persist_dir=memory_file)
                    self.index = load_index_from_storage(storage_context)
                    
                    # Create query engine
                    self.query_engine = self.index.as_query_engine()
                    
                    logger.info(f"Successfully loaded existing memory index for user {self.user_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error loading existing memory index: {e}")
                    logger.info("Will create a new memory index")
            
            # Load transcripts if not already loaded
            if not self.transcripts:
                success = self.load_transcripts()
                if not success:
                    logger.error(f"No transcripts found for user {self.user_id}")
                    return False
            
            # Configure settings with OpenAI embeddings and no LLM
            embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            Settings.embed_model = embed_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            Settings.llm = None  # Explicitly disable LLM component
            
            logger.info(f"Creating memory index for user {self.user_id}")
            
            # Process transcripts into documents
            documents = self._prepare_documents()
            
            if not documents:
                logger.warning(f"No documents created from transcripts for user {self.user_id}")
                return False
            
            logger.info(f"Created {len(documents)} memory documents from transcripts")
            
            # Create and save index
            start_time = time.time()
            
            # Set up storage
            storage_context = StorageContext.from_defaults()
            
            # Create the index
            nodes = documents
            
            # Load and set the index using Pinecone
            logger.info(f"Creating LlamaIndex from nodes with Pinecone for user {self.user_id}")
            
            try:
                # Get or create Pinecone index for user's transcripts
                self.pinecone_index_name = pinecone_manager.get_transcript_index(self.user_id)
                logger.info(f"Using Pinecone index for transcripts: {self.pinecone_index_name}")
                
                # Import Pinecone vector store with correct path
                try:
                    from llama_index.vector_stores.pinecone import PineconeVectorStore
                except ImportError:
                    # Try alternate import path
                    from llama_index.core.vector_stores.pinecone import PineconeVectorStore
                
                # Clear existing vectors for a fresh start
                pinecone_manager.clear_index(self.pinecone_index_name)
                
                # Get Pinecone index
                pinecone_index = pinecone_manager.get_index(self.pinecone_index_name)
                
                # Create vector store with Pinecone
                vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
                
                # Create vector index with Pinecone
                self.index = VectorStoreIndex(nodes, 
                                              service_context=storage_context,
                                              vector_store=vector_store)
                                              
                logger.info(f"Successfully stored user transcript vectors in Pinecone index: {self.pinecone_index_name}")
            except Exception as e:
                logger.error(f"Error using Pinecone for user transcripts, falling back to in-memory: {e}")
                # Fallback to in-memory storage
                self.index = VectorStoreIndex(nodes, service_context=storage_context)
            
            # Create optimized query engine
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.response_synthesizers import get_response_synthesizer
            
            # Create optimized retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3  # Limit results for faster retrieval
            )
            
            # Create response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",  # More efficient response generation
                use_async=True  # Use async for better performance
            )
            
            # Create optimized query engine
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )
            
            logger.info(f"Memory index created and saved in {time.time() - start_time:.2f} seconds")
            return True
            
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            logger.error("Make sure llama-index and openai are installed")
            return False
        except Exception as e:
            logger.error(f"Error creating memory index: {e}")
            return False
    
    def _prepare_documents(self) -> List[Any]:
        """
        Process transcripts into documents for indexing
        
        Returns:
            List of Document objects
        """
        from llama_index.core import Document
        
        documents = []
        
        # Process each transcript
        for i, transcript in enumerate(self.transcripts):
            try:
                # Extract metadata using the structured format if available
                session_id = transcript.get('session_id', f"session_{i}")
                
                # Handle different transcript formats (structured vs legacy)
                session_date = None
                topic_summary = None
                therapist_name = None
                client_name = None
                
                # Check for structured session info
                if 'session_info' in transcript:
                    session_info = transcript['session_info']
                    session_date = session_info.get('date', '')
                    topic_summary = session_info.get('topic_summary', '')
                
                # Check for structured therapist/client info
                if 'therapist' in transcript:
                    therapist_name = transcript['therapist'].get('name', '')
                
                if 'client' in transcript:
                    client_name = transcript['client'].get('name', '')
                
                # Fall back to legacy fields if needed
                if not session_date:
                    session_date = transcript.get('created_at', transcript.get('processed_at', ''))
                
                # Process transcript utterances
                utterances = transcript.get('transcript', [])
                if not utterances:
                    # Check if it's using a different field name
                    utterances = transcript.get('utterances', [])
                
                if not utterances:
                    continue
                
                # Determine speakers if not already identified
                if not therapist_name or not client_name:
                    speakers = set(utterance.get('speaker', '') for utterance in utterances)
                    
                    # Try to identify therapist and client
                    therapist_indicators = ['dr', 'doctor', 'therapist', 'counselor']
                    for speaker in speakers:
                        speaker_lower = speaker.lower()
                        if any(indicator in speaker_lower for indicator in therapist_indicators):
                            therapist_name = speaker
                            # Client is likely the other speaker
                            for other_speaker in speakers:
                                if other_speaker != speaker:
                                    client_name = other_speaker
                                    break
                            break
                    
                    # If still not identified, make a best guess
                    if not therapist_name and len(speakers) >= 2:
                        # First speaker is often the therapist
                        therapist_name = utterances[0].get('speaker', '')
                        # Find the first different speaker for client
                        for utterance in utterances:
                            if utterance.get('speaker') != therapist_name:
                                client_name = utterance.get('speaker', '')
                                break
                
                # Group utterances into conversation chunks with context
                chunk_size = 5  # Number of utterances per chunk
                overlap = 2     # Overlap between chunks
                
                for j in range(0, len(utterances), chunk_size - overlap):
                    chunk_utterances = utterances[j:j + chunk_size]
                    
                    # Format chunk content with enhanced metadata
                    content = f"Session: {session_id}\n"
                    
                    if session_date:
                        # Format date if it's ISO format
                        try:
                            if isinstance(session_date, str) and 'T' in session_date:
                                display_date = session_date.split('T')[0]
                            else:
                                display_date = str(session_date)
                        except Exception as e:
                            logger.warning(f"Error formatting date {session_date}: {e}")
                            display_date = str(session_date)
                            
                        content += f"Date: {display_date}\n"
                    
                    # Add therapist and client information
                    if therapist_name:
                        content += f"Therapist: {therapist_name}\n"
                    if client_name:
                        content += f"Client: {client_name}\n"
                    
                    # Add topic summary if available
                    if topic_summary:
                        content += f"Topic: {topic_summary}\n"
                    
                    content += "\nConversation:\n"
                    
                    # Add conversation
                    for utterance in chunk_utterances:
                        speaker = utterance.get('speaker', 'Unknown')
                        text = utterance.get('text', '')
                        if text:
                            content += f"{speaker}: {text}\n"
                    
                    # Add emotion analysis if available
                    if 'emotion_analysis' in transcript:
                        # Find emotion analysis for this chunk
                        for utterance in chunk_utterances:
                            text = utterance.get('text', '')
                            speaker = utterance.get('speaker', '')
                            
                            # Look for matching emotion analysis
                            for emotion_data in transcript.get('emotion_analysis', []):
                                if emotion_data.get('text') == text and emotion_data.get('speaker') == speaker:
                                    # Add emotion data
                                    emotions = emotion_data.get('predicted_wav_emotion', emotion_data.get('emotions', []))
                                    if emotions:
                                        if isinstance(emotions, list):
                                            emotion_str = ", ".join(emotions)
                                        else:
                                            emotion_str = str(emotions)
                                        content += f"  Emotion: {emotion_str}\n"
                    
                    # Create document with metadata
                    doc = Document(
                        text=content,
                        metadata={
                            "session_id": session_id,
                            "client": client_name or self.user_id,
                            "therapist": therapist_name or "Unknown",
                            "chunk": j,
                            "date": str(session_date)
                        }
                    )
                    
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing transcript {i}: {e}")
        
        return documents
    def query_memory(self, query_text: str, max_results: int = 3) -> str:
        """
        Query user's therapy memory with a specific question
        
        Args:
            query_text: The question or query about past therapy sessions
            max_results: Maximum number of memory chunks to return
            
        Returns:
            Relevant information from past therapy sessions
        """
        if not self.query_engine:
            if not self.create_memory_index():
                return "Sorry, I couldn't access your therapy session history."
        
        try:
            # Check cache first (normalize query for cache key)
            cache_key = query_text.strip().lower()
            if cache_key in self.query_cache:
                self.cache_hits += 1
                result = self.query_cache[cache_key]
                logger.info(f"Memory query answered from cache for user {self.user_id} (hits: {self.cache_hits}, misses: {self.cache_misses})")
                return result
                
            # Cache miss - execute query
            self.cache_misses += 1
            logger.info(f"Querying memory for user {self.user_id}: {query_text}")
            start_time = time.time()
            
            # Execute query
            response = self.query_engine.query(query_text)
            
            logger.info(f"Memory query answered in {time.time() - start_time:.2f} seconds")
            
            # Format response for readability
            result = self._format_memory_response(str(response))
            
            # Update cache (simple LRU implementation)
            if len(self.query_cache) >= self.cache_size:
                # Remove oldest item
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
                
            # Add to cache
            self.query_cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            return f"I had trouble recalling that information. Error: {str(e)}"
    
    def _format_memory_response(self, response: str) -> str:
        """Format the memory response for better readability"""
        # Extract the actual content if using LlamaIndex's standard format
        if "Context information is below." in response:
            parts = response.split("Context information is below.")
            if len(parts) > 1:
                content_parts = parts[1].split("Given the context information and not prior knowledge, answer the query.")
                if len(content_parts) > 0:
                    response = content_parts[0].strip()
                    response = response.replace("---------------------", "")
        
        # Clean up file paths for better readability
        response = re.sub(r'file_path: .*?therapy_documents', 'From session', response)
        
        # Format nicely
        formatted = "THERAPY SESSION MEMORY:\n\n"
        formatted += response
        
        return formatted
    
    def extract_memories_for_letta(self, message: str) -> str:
        """
        Extract relevant memories based on the user message for Letta
        
        This method analyzes a user message and retrieves relevant therapy
        memories that can enhance Letta's response.
        
        Args:
            message: User message to analyze for memory retrieval
            
        Returns:
            Relevant memories formatted for Letta prompt enhancement
        """
        # Extract potential topics or questions from the message
        topics = self._extract_therapy_topics(message)
        
        if not topics:
            return ""
        
        # Query each topic and combine results
        all_memories = ""
        for topic in topics[:2]:  # Limit to top 2 topics to avoid too much context
            memory = self.query_memory(topic)
            if memory and len(memory) > 20:  # Check if meaningful results
                if all_memories:
                    all_memories += "\n\n---\n\n"
                all_memories += memory
        
        return all_memories
    
    def _extract_therapy_topics(self, message: str) -> List[str]:
        """Extract potential therapy topics from a message"""
        topics = []
        
        # Look for direct questions
        questions = re.findall(r'[^.!?]*\?', message)
        topics.extend([q.strip() for q in questions if len(q.strip()) > 10])
        
        # Extract key phrases with therapy-related keywords
        therapy_keywords = [
            'remember', 'mentioned', 'talked about', 'said', 'told', 'discussed',
            'session', 'last time', 'therapy', 'feeling', 'felt', 'emotion',
            'progress', 'goal', 'plan', 'problem', 'issue'
        ]
        
        for keyword in therapy_keywords:
            if keyword in message.lower():
                # Extract the sentence containing the keyword
                sentences = re.split(r'[.!?]+', message)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        clean = sentence.strip()
                        if clean and len(clean) > 10 and clean not in topics:
                            topics.append(clean)
        
        # If still no topics found, use the whole message
        if not topics and len(message) > 5:
            topics.append(message)
            
        return topics


# End of get_conversation_topics method
    
    def enhance_prompt(self, message: str, system_prompt: str) -> str:
        """
        Enhance a Letta system prompt with relevant therapy memories
        
        Args:
            message: The user message
            system_prompt: The original system prompt
            
        Returns:
            Enhanced system prompt with relevant therapy memories
        """
        if not self.initialized:
            return system_prompt
        
        # Extract relevant memories
        memories = self.memory.extract_memories_for_letta(message)
        
        if not memories:
            return system_prompt
        
        # Add memories to system prompt
        enhanced_prompt = system_prompt + "\n\n"
        enhanced_prompt += memories
        
        return enhanced_prompt
    
    def get_letta_response(self, client, agent_id: str, message: str, system_prompt: str = None):
        """
        Get a response from a Letta agent with enhanced therapy memories
        
        Args:
            client: Letta client instance
            agent_id: ID of the Letta agent
            message: User message
            system_prompt: Optional system prompt to use
            
        Returns:
            Agent response with enhanced therapy memories
        """
        try:
            # If we have a system prompt and memory is initialized, enhance it
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
            return f"I'm having trouble connecting to your therapy history. Error: {e}"


def main():
    """Test the therapy transcript memory system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Therapy Transcript Memory')
    parser.add_argument('--user_id', type=str, required=True, help='User ID to retrieve memories for')
    parser.add_argument('--query', type=str, help='Query to test memory retrieval')
    parser.add_argument('--reload', action='store_true', help='Force reload memory index')
    args = parser.parse_args()
    
    # Initialize memory for the specified user
    memory = TherapyTranscriptMemory(args.user_id)
    
    # Load transcripts
    logger.info(f"Loading transcripts for user {args.user_id}")
    success = memory.load_transcripts()
    if not success:
        logger.error(f"No transcripts found for user {args.user_id}")
        return
    
    # Create memory index
    logger.info(f"Creating memory index for user {args.user_id}")
    success = memory.create_memory_index(force_reload=args.reload)
    if not success:
        logger.error(f"Failed to create memory index for user {args.user_id}")
        return
    
    # Test memory retrieval if query provided
    if args.query:
        logger.info(f"Testing memory retrieval with query: {args.query}")
        result = memory.query_memory(args.query)
        print("\nMEMORY RETRIEVAL RESULT:")
        print("-" * 40)
        print(result)
    else:
        # Example queries if none provided
        test_queries = [
            "What did I say about my anxiety in past sessions?",
            "What have we discussed about my goals?",
            "What symptoms have I mentioned before?",
            "What progress have I made in therapy?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 40)
            
            start_time = time.time()
            result = memory.query_memory(query)
            print(f"Retrieved in {time.time() - start_time:.2f} seconds")
            
            print(result)
            print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # If run without arguments, print usage
        print("Usage: python transcript_memory.py --user_id USER_ID [--query QUERY] [--reload]")
        print("\nExample:")
        print("  python transcript_memory.py --user_id Dr.Smith --query \"What did I say about anxiety?\"")
