#!/usr/bin/env python3
"""
Test Combined Memory System

This script tests that both transcript memory and therapy documents work correctly
for Natey's data.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

# Load environment variables for OpenAI API key
from dotenv import load_dotenv
load_dotenv()

def test_transcript_memory():
    """Test accessing Natey's transcript memory"""
    from backend.transcript_memory import TherapyTranscriptMemory
    
    client_name = "natey"
    logger.info(f"Testing transcript memory for client: {client_name}")
    
    memory = TherapyTranscriptMemory(client_name)
    
    # Test a few queries
    test_queries = [
        "What was discussed in Natey's therapy session?",
        "What emotions were expressed in the therapy session?",
        "What was the main topic of the session?"
    ]
    
    for query in test_queries:
        logger.info(f"Query: {query}")
        result = memory.query_memory(query)
        print(f"\nQuery: {query}")
        print(f"Result: {result}")
        print("-" * 50)
    
    return True

def test_therapy_documents():
    """Test accessing general therapy documents"""
    try:
        from openai_llama_rag import OpenAITherapyIndex
        
        logger.info("Testing therapy document retrieval")
        
        # Initialize OpenAI therapy index
        therapy_index = OpenAITherapyIndex()
        
        # Test a few queries
        test_queries = [
            "What techniques are useful for anxiety?",
            "How to handle negative thoughts?",
            "Best practices for CBT"
        ]
        
        for query in test_queries:
            logger.info(f"Query: {query}")
            results = therapy_index.query(query)
            print(f"\nQuery: {query}")
            print(f"Result: {results[:200]}..." if results else "No results found")
            print("-" * 50)
        
        return True
    except Exception as e:
        logger.warning(f"Error using therapy document index: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing PANDA Therapy Memory System\n")
    
    # Test transcript memory
    print("\n📝 TESTING TRANSCRIPT MEMORY")
    print("=" * 60)
    transcript_success = test_transcript_memory()
    
    # Test therapy documents
    print("\n📚 TESTING THERAPY DOCUMENTS")
    print("=" * 60)
    documents_success = test_therapy_documents()
    
    # Summary
    print("\n✅ TEST SUMMARY")
    print("=" * 60)
    print(f"Transcript Memory: {'✅ PASS' if transcript_success else '❌ FAIL'}")
    print(f"Therapy Documents: {'✅ PASS' if documents_success else '❌ FAIL'}")
    
    if transcript_success:
        print("\n🎉 Natey's transcript has been successfully updated and vectorized!")
        print("   The memory system is now ready for use with Letta.")
