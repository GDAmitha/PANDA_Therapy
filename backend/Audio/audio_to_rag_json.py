import whisper
import json
import torchaudio
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from pydub import AudioSegment
import uuid
import os

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# Load emotion classification model (fine-tuned on MSP-Podcast or similar)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

def transcribe_audio(path):
    result = whisper_model.transcribe(path)
    return result["text"], result["segments"]

def classify_emotion(path):
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.squeeze().numpy()
    inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=1).item()
    emotion_labels = emotion_model.config.id2label
    return emotion_labels[predicted_id]

def convert_mp3_to_wav(mp3_path):
    audio = AudioSegment.from_file(mp3_path)
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

def generate_rag_json(audio_path, output_json_path):
    if audio_path.endswith(".mp3"):
        audio_path = convert_mp3_to_wav(audio_path)
    
    transcript, segments = transcribe_audio(audio_path)
    emotion = classify_emotion(audio_path)
    
    chunk = {
        "id": str(uuid.uuid4()),
        "summary": transcript[:400],  # rough summary for now
        "emotion": emotion,
        "topics": [],  # for future tagging
        "source": os.path.basename(audio_path),
        "timestamp": "0-5min",
        "speaker": "client",
        "embedding": None  # to fill during vector indexing
    }

    with open(output_json_path, "w") as f:
        json.dump(chunk, f, indent=4)
    
    return chunk

# Example use:
# audio_file = "path/to/your_audio.wav"
# output_file = "output/session_summary.json"
# print(generate_rag_json(audio_file, output_file))