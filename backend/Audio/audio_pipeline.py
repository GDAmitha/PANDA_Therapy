## DEPRECIATED ###
import os
import uuid
import time
import tempfile
import whisper
from collections import Counter

from dotenv import load_dotenv
import whisper
import torch
import torchaudio
from pydub import AudioSegment
# from pyannote.audio import Pipeline
from transformers import (
    pipeline as hf_pipeline,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor
)
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Temporarily commenting out model loading
# whisper_model = whisper.load_model("base")
# diariation_pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization",
#     use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
# )
# emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
#     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# )
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
#     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# )
# sentiment_pipeline = hf_pipeline("sentiment-analysis")
# topic_pipeline = hf_pipeline(
#     "zero-shot-classification",
#     model="facebook/bart-large-mnli"
# )
# dialogue_act_pipeline = hf_pipeline(
#     "text-classification",
#     model="facebook/bart-large-mnli"
# )
# summarization_pipeline = hf_pipeline(
#     "summarization",
#     model="facebook/bart-large-cnn"
# )

embeddings = OpenAIEmbeddings()

# Temporarily commenting out functions that require the models
# def classify_emotion(path: str) -> str:
#     """
#     Classify the overall emotion of an audio file segment.
#     """
#     waveform, sample_rate = torchaudio.load(path)
#     waveform = waveform.squeeze().numpy()
#     inputs = feature_extractor(
#         waveform,
#         sampling_rate=sample_rate,
#         return_tensors="pt",
#         padding=True
#     )
#     with torch.no_grad():
#         logits = emotion_model(**inputs).logits
#         predicted_id = torch.argmax(logits, dim=1).item()
#     return emotion_model.config.id2label[predicted_id]

def process_audio_session(audio_path: str) -> dict:
    """
    Placeholder function that returns an empty session
    """
    return {
        "session_id": f"{int(time.time())}",
        "segments": []
    }