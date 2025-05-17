"""
Compatibility layer for audio processing components
This file provides a centralized way to import all audio processing components
"""
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the Audio directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Export all components needed for audio processing
try:
    # Import the transcription module
    from .transcribe import transcribe_audio
    from .transcribe_to_json import transcribe_audio_to_json
    
    # Import the speaker role assignment module
    from .assign_speaker_roles import assign_speaker_roles
    
    # Import the emotion analysis module
    from .assign_audio_emotions import assign_audio_emotions
    
    # Import the Letta analysis module
    from .letta_analyze import TherapySessionAnalyzer, analyze_therapy_session
    
    # Import the audio processing pipeline
    from .audio_process import process_audio_file, process_audio_file_fallback
    
    # Check if Assembly AI API key is available
    assembly_api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not assembly_api_key:
        logger.warning("ASSEMBLYAI_API_KEY not found in environment variables")
    else:
        logger.info("ASSEMBLYAI_API_KEY is available")
    
    # Flag indicating that all components are available
    AUDIO_PROCESSING_AVAILABLE = True
    logger.info("All audio processing components loaded successfully")
    
except ImportError as e:
    logger.error(f"Failed to import audio processing components: {str(e)}")
    AUDIO_PROCESSING_AVAILABLE = False
except Exception as e:
    logger.error(f"Unexpected error initializing audio modules: {str(e)}")
    AUDIO_PROCESSING_AVAILABLE = False

def is_audio_processing_available():
    """Check if audio processing is available"""
    return AUDIO_PROCESSING_AVAILABLE

def initialize_therapy_analyzer():
    """Initialize the TherapySessionAnalyzer if available"""
    if AUDIO_PROCESSING_AVAILABLE:
        try:
            return TherapySessionAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize TherapySessionAnalyzer: {str(e)}")
    return None
