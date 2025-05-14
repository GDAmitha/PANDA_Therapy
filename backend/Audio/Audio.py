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
)
from pyannote.audio import Pipeline
from pyannote.core import Annotation

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


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
    Assign each word-level segment to the speaker turn with the
    maximum overlap.
    word_segments: list of {'timestamp': (start, end), 'text': str}
    diarization: pyannote.core.Annotation
    Returns list of {'start','end','speaker','text'}
    """
    transcript = []
    for chunk in word_segments:
        start, end = chunk.get("timestamp", (None, None))
        if start is None or end is None:
            continue
        # compute overlaps
        overlaps = []
        for turn, _, label in diarization.itertracks(yield_label=True):
            ovl = max(0, min(turn.end, end) - max(turn.start, start))
            if ovl > 0:
                overlaps.append((ovl, label))
        if overlaps:
            speaker = max(overlaps, key=lambda x: x[0])[1]
        else:
            speaker = "unknown"
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
    gap_tolerance: max gap (seconds) between words to merge.
    Returns list of {'start','end','speaker','text'} per turn.
    """
    turns = []
    for seg in word_segments:
        if not turns:
            turns.append(seg.copy())
            continue
        prev = turns[-1]
        same_speaker = seg["speaker"] == prev["speaker"]
        small_gap = (seg["start"] - prev["end"]) <= gap_tolerance
        if same_speaker and small_gap:
            prev["text"] += " " + seg["text"]
            prev["end"] = seg["end"]
        else:
            turns.append(seg.copy())
    return turns


def transcribe_and_segment(audio_path: str, asr_model: str = "openai/whisper-small") -> list[dict]:
    """
    Full pipeline: load/audio
di arize
diarization
transcribe words
assign speakers
merge turns
    """
    # 1) load & resample
    # we know WhisperFeatureExtractor expects 16 kHz
    processor = WhisperProcessor.from_pretrained(asr_model)
    #target_sr = processor.feature_extractor.sampling_rate
    #speech, sr = load_and_resample(audio_path, target_sr)

    # 2) run diarization
    diarization = diarize(audio_path, HF_TOKEN)

    # 3) prepare ASR pipeline for full-file transcription
    model = WhisperForConditionalGeneration.from_pretrained(asr_model)
    # force English transcription
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

    # 4) run ASR word-level
    chunks = asr_pipe(audio_path).get("chunks", [])
    #chunks = asr_pipe({"array": speech, "sampling_rate": sr}).get("chunks", [])
    # filter out zero-length or duplicate chunks
    #chunks = [c for c in raw_chunks if c.get("timestamp") and c["timestamp"][1] > c["timestamp"][0]]
    # 5) assign speakers to words
    word_segments = assign_speakers(chunks, diarization)

    # 6) merge into turns
    turn_segments = merge_speaker_turns(word_segments)
    return turn_segments