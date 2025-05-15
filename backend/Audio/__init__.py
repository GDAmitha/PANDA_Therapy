"""
Audio processing library for therapy sessions
Provides tools for transcription, speaker identification, and emotion analysis
"""

from .modules.pipeline import TherapyAudioPipeline, process_audio_session
from .modules.transcriber import transcribe_audio, transcribe_audio_to_json
from .modules.speaker_assignment import assign_speaker_roles
from .modules.emotion_analyzer import EmotionAnalyzer, analyze_audio_emotions

__all__ = [
    'TherapyAudioPipeline',
    'process_audio_session',
    'transcribe_audio',
    'transcribe_audio_to_json',
    'assign_speaker_roles',
    'EmotionAnalyzer',
    'analyze_audio_emotions'
]
