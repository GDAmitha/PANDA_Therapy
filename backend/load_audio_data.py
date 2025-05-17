"""
Script to load existing audio emotion analysis data into the RAG system
"""
import os
import logging
from rag_agent import TherapyRAGAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_audio_emotion_files():
    """Load all audio emotion JSON files in the audio_emo_transcript directory into the RAG system."""
    try:
        # Initialize the RAG agent
        agent = TherapyRAGAgent()
        
        # Load the regular therapy documents first
        logger.info("Loading therapy documents...")
        agent.load_documents("../therapy_documents")
        
        # Load the audio emotion files
        audio_emo_dir = "./Audio/audio_emo_transcript"
        if not os.path.exists(audio_emo_dir):
            logger.warning(f"Audio emotion directory not found: {audio_emo_dir}")
            return False
        
        files_loaded = 0
        for filename in os.listdir(audio_emo_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(audio_emo_dir, filename)
                logger.info(f"Loading audio emotion file: {file_path}")
                
                # Load the audio data into a separate index
                agent.load_audio_data(file_path)
                files_loaded += 1
        
        if files_loaded > 0:
            logger.info(f"Successfully loaded {files_loaded} audio emotion files")
            return True
        else:
            logger.warning("No audio emotion files found to load")
            return False
            
    except Exception as e:
        logger.error(f"Error loading audio emotion files: {str(e)}")
        return False

if __name__ == "__main__":
    success = load_audio_emotion_files()
    if success:
        print("Audio emotion data successfully loaded into the RAG system")
    else:
        print("Failed to load audio emotion data")
