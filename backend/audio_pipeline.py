import os
import uuid
import time
import tempfile
from collections import Counter

from dotenv import load_dotenv
import whisper
import torch
import torchaudio
from pydub import AudioSegment
from pyannote.audio import Pipeline
from transformers import (
    pipeline as hf_pipeline,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor
)
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# 1) Load models & pipelines
whisper_model = whisper.load_model("base")

diariation_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
)

emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

sentiment_pipeline = hf_pipeline("sentiment-analysis")
topic_pipeline = hf_pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
dialogue_act_pipeline = hf_pipeline(
    "text-classification",
    model="mrm8488/distilroberta-finetuned-dialog-act"
)
summarization_pipeline = hf_pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

embeddings = OpenAIEmbeddings()


def classify_emotion(path: str) -> str:
    """
    Classify the overall emotion of an audio file segment.
    """
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.squeeze().numpy()
    inputs = feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=1).item()
    return emotion_model.config.id2label[predicted_id]


def process_audio_session(audio_path: str) -> dict:
    """
    Process an entire therapy audio file into a detailed RAG JSON structure.
    Returns a dict: { session_id: str, segments: List[dict] }
    """
    # 1) Speaker diarization
    diarization = diariation_pipeline({"audio": audio_path})
    audio = AudioSegment.from_file(audio_path)

    segments = []
    for turn, track in diarization.itertracks(yield_label=True):
        start_sec, end_sec = turn.start, turn.end
        speaker = track

        # Extract segment audio
        segment_audio = audio[start_sec * 1000: end_sec * 1000]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            seg_path = tmp.name
            segment_audio.export(seg_path, format="wav")

        # 2) Transcription
        whisper_res = whisper_model.transcribe(seg_path)
        transcript = whisper_res.get("text", "")

        # 3) Emotion & Sentiment
        emotion = classify_emotion(seg_path)
        sent = sentiment_pipeline(transcript)[0]
        sentiment_score = sent.get("score")

        # 4) Topic tagging
        topics = topic_pipeline(
            transcript,
            candidate_labels=[
                "anxiety", "depression", "relationships",
                "sleep", "stress", "self-esteem"
            ]
        ).get("labels", [])

        # 5) Dialogue act
        dialogue_act = dialogue_act_pipeline(transcript)[0]["label"]

        # 6) Summarization
        summary = summarization_pipeline(
            transcript,
            max_length=60,
            min_length=20
        )[0]["summary_text"]

        # 7) Keywords (top 5 tokens >3 chars)
        words = [w.lower().strip(".,!?;:") for w in transcript.split() if len(w) > 3]
        freq = Counter(words)
        keywords = [tok for tok, _ in freq.most_common(5)]

        # 8) Embedding
        embedding = embeddings.embed_documents([transcript])[0]

        segments.append({
            "id": str(uuid.uuid4()),
            "speaker": speaker,
            "start_time": start_sec,
            "end_time": end_sec,
            "transcript": transcript,
            "summary": summary,
            "emotion": emotion,
            "sentiment_score": sentiment_score,
            "topics": topics,
            "dialogue_act": dialogue_act,
            "keywords": keywords,
            "embedding": embedding
        })

    return {
        "session_id": f"{int(time.time())}",
        "segments": segments
    }