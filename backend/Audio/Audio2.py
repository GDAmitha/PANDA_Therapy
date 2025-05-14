import os
from dotenv import load_dotenv
import torch
import soundfile as sf
import numpy as np
import librosa
from transformers import (
    pipeline,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)
from pyannote.audio import Pipeline
from pyannote.core import Annotation

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# -------------------------
# emotion classification integration using Wav2Vec2
# -------------------------
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
emotion_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
emotion_model.eval()


def load_and_resample(audio_path: str, target_sr: int):
    """
    Load an audio file and resample to target sampling rate (e.g., 16000).
    Returns mono waveform and sampling rate.
    """
    speech, sr = sf.read(audio_path)
    # to mono
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)
    # resample if needed
    if sr != target_sr:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return speech, sr


def diarize(audio_path: str, hf_token: str, speaker_model: str = "pyannote/speaker-diarization-3.1"):
    """
    Perform speaker diarization with exactly two speakers.
    Returns a pyannote.core.Annotation.
    """
    pipeline = Pipeline.from_pretrained(
        speaker_model,
        use_auth_token=hf_token
    )
    return pipeline(audio_path, num_speakers=2)


def assign_speakers(word_segments: list[dict], diarization: Annotation):
    """
    Assign each word-level segment to the speaker turn with the maximum overlap.
    Returns list of {'start','end','speaker','text'}
    """
    transcript = []
    for chunk in word_segments:
        start, end = chunk.get("timestamp", (None, None))
        if start is None or end is None:
            continue
        overlaps = []
        for turn, _, label in diarization.itertracks(yield_label=True):
            ovl = max(0, min(turn.end, end) - max(turn.start, start))
            if ovl > 0:
                overlaps.append((ovl, label))
        speaker = max(overlaps, key=lambda x: x[0])[1] if overlaps else "unknown"
        transcript.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": chunk.get("text", "").strip()
        })
    return transcript


def merge_speaker_turns(word_segments: list[dict], gap_tolerance: float = 0.5):
    """
    Merge consecutive same-speaker word segments into full turns.
    Returns list of {'start','end','speaker','text'} per turn.
    """
    turns = []
    for seg in word_segments:
        if not turns:
            turns.append(seg.copy())
            continue
        prev = turns[-1]
        if seg["speaker"] == prev["speaker"] and (seg["start"] - prev["end"]) <= gap_tolerance:
            prev["text"] += " " + seg["text"]
            prev["end"] = seg["end"]
        else:
            turns.append(seg.copy())
    return turns


def extract_emotion_label(waveform: np.ndarray, sr: int) -> dict:
    """
    Given a waveform and its sample rate, classify emotion with Wav2Vec2.
    Returns a dict with 'label' and 'score'.
    """
    # prepare input
    inputs = emotion_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    labels = emotion_model.config.id2label
    # pick top
    idx = int(np.argmax(probs))
    return {"label": labels[idx], "score": probs[idx]}


def transcribe_and_segment(audio_path: str, asr_model: str = "openai/whisper-small") -> list[dict]:
    """
    Full pipeline:
      1) load & resample
      2) diarize
      3) ASR word-level
      4) assign speakers
      5) merge turns
      6) classify emotion per turn
    """
    # 1) load & resample for emotion classification
    processor = WhisperProcessor.from_pretrained(asr_model)
    target_sr = processor.feature_extractor.sampling_rate
    speech, sr = load_and_resample(audio_path, target_sr)

    # 2) run diarization
    diarization = diarize(audio_path, HF_TOKEN)

    # 3) prepare ASR pipeline
    model = WhisperForConditionalGeneration.from_pretrained(asr_model)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )
    device = 0 if torch.cuda.is_available() else -1
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=[5, 5],
        return_timestamps="word",
        device=device,
    )

    # 4) ASR word-level
    raw = asr_pipe(audio_path)
    chunks = raw.get("chunks", [])

    # 5) assign speakers
    word_segments = assign_speakers(chunks, diarization)

    # 6) merge into turns
    turn_segments = merge_speaker_turns(word_segments)

    # 7) classify emotion for each turn
    for turn in turn_segments:
        start_sample = int(turn['start'] * sr)
        end_sample = int(turn['end'] * sr)
        segment_waveform = speech[start_sample:end_sample]
        emo = extract_emotion_label(segment_waveform, sr)
        turn['emotion'] = emo

    return turn_segments
