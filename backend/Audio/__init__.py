"""
Audio processing library for therapy sessions
Provides tools for transcription, speaker identification, and emotion analysis
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use our compatibility layer to handle all imports
from .audio_compat import (
    is_audio_processing_available,
    initialize_therapy_analyzer,
    AUDIO_PROCESSING_AVAILABLE,
)

# Export all components from audio_compat
if AUDIO_PROCESSING_AVAILABLE:
    try:
        # Import and export core processing components
        from .audio_process import process_audio_file, process_audio_file_fallback
        from .transcribe import transcribe_audio
        from .transcribe_to_json import transcribe_audio_to_json
        from .assign_speaker_roles import assign_speaker_roles
        from .assign_audio_emotions import assign_audio_emotions, extract_emotions_from_keywords
        from .letta_analyze import TherapySessionAnalyzer, analyze_therapy_session
        
        # Try importing optional modules components if available
        try:
            from .modules.pipeline import TherapyAudioPipeline, process_audio_session
            from .modules.transcriber import transcribe_audio as module_transcribe_audio
            from .modules.speaker_assignment import assign_speaker_roles as module_assign_roles
            from .modules.emotion_analyzer import EmotionAnalyzer, analyze_audio_emotions as module_analyze_emotions
        except ImportError as e:
            logger.warning(f"Some module components not available: {str(e)}")
            
    except ImportError as e:
        logger.error(f"Failed to import core audio components: {str(e)}")
        AUDIO_PROCESSING_AVAILABLE = False

__all__ = [
    # Main processing functions
    'process_audio_file',
    'process_audio_file_direct',
    'process_audio_file_fallback',
    'TherapySessionAnalyzer',
    'analyze_therapy_session',
    
    'TherapyAudioPipeline',
    'process_audio_session',
    'transcribe_audio',
    'transcribe_audio_to_json',
    'assign_speaker_roles',
    'EmotionAnalyzer',
    'analyze_audio_emotions'
]
