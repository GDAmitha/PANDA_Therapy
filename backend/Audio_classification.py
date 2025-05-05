## DEPRECIATED ###
import os
import argparse
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.core import Annotation
import soundfile as sf
import numpy as np

load_dotenv()

# hf_token = os.getenv("HF_TOKEN")

def diarize(audio_path: str, hf_token: str, speaker_model: str):
    """
    Perform speaker diarization using a Hugging Face transformers pipeline.
    Returns a list of segments: [{'start': float, 'end': float, 'label': str}, ...].
    """
    diarization_pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    return diarization_pipe(audio_path, num_speakers=2)





def transcribe(audio_path, diarization, asr_model="openai/whisper-small"):
    # load the audio once
    speech, sr = sf.read(audio_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load processor + model, force English transcribe
    processor = WhisperProcessor.from_pretrained(asr_model)
    model     = WhisperForConditionalGeneration.from_pretrained(asr_model).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )

    results = []
    # iterate over each (segment, _, label)
    for segment, _, label in diarization.itertracks(yield_label=True):
        # extract the slice of the waveform
        start_frame = int(segment.start * sr)
        end_frame   = int(segment.end   * sr)
        chunk       = speech[start_frame:end_frame]

        # preprocess & forward
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        input_feats = inputs.input_features.to(device)

        predicted_ids = model.generate(input_feats)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        results.append({
            "speaker": label,
            "start":   float(segment.start),
            "end":     float(segment.end),
            "text":    text
        })

    return results


# def transcribe(audio_path: str, asr_model: str):
#     """
#     Transcribe audio using a Hugging Face ASR pipeline.
#     Returns a list of segments: [{'timestamp': (start, end), 'text': str}, ...].
#     """
    
#     asr_model = "openai/whisper-small"
#     processor = WhisperProcessor.from_pretrained(asr_model)
#     model = WhisperForConditionalGeneration.from_pretrained(asr_model)
#     model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
#         language = "en",
#         task = "transcribe"
#     )

#     info = sf.info(audio_path)
#     duration = info.frames / info.samplerate 

#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     asr_pipe = pipeline(
#         "automatic-speech-recognition",
#         model= model,
#         #tokenizer = processor.tokenizer,
#         #fearture_extractor = processor.feature_extractor,
#         chunk_length_s= duration,
#         stride_length_s=[0, 0],
#         return_timestamps="word",
#         device=device,
#     )
#     result = asr_pipe(audio_path)
#     return result.get("chunks", [])

def assign_speakers(asr_chunks, diarization: Annotation):
    """
    asr_chunks:  list of dicts from HF ASR pipeline, each with
                 - 'timestamp': (start_sec, end_sec)
                 - 'text': str
    diarization: pyannote.core.Annotation from your diarization pipeline
    """
    transcript = []
    for chunk in asr_chunks:
        # 1) get start/end from 'timestamp' (or fallback if you ever have start/end keys)
        if "timestamp" in chunk:
            start, end = chunk["timestamp"]
        else:
            start, end = chunk.get("start", 0.0), chunk.get("end", 0.0)

        if start is None or end is None:
            continue

        # 2) find which speaker turn covers the midpoint
        mid = (start + end) / 2
        speaker = "unknown"
        for turn, _, label in diarization.itertracks(yield_label=True):
            if turn.start <= mid <= turn.end:
                speaker = label
                break

        # 3) append normalized record
        transcript.append({
            "start": start,
            "end":   end,
            "speaker": speaker,
            "text":  chunk.get("text", "").strip()
        })

    return transcript

def merge_speaker_turns(word_segments, gap_tolerance=1.0):
    """
    word_segments: list of dicts with start,end,speaker,text
    gap_tolerance: max seconds between words to still merge
    """
    turns = []
    for seg in word_segments:
        if not turns:
            turns.append(seg.copy())
            continue

        prev = turns[-1]
        same_speaker = seg["speaker"] == prev["speaker"]
        small_gap    = (seg["start"] - prev["end"]) <= gap_tolerance

        if same_speaker and small_gap:
            # extend previous turn
            prev["text"] += " " + seg["text"]
            prev["end"]   = seg["end"]
        else:
            turns.append(seg.copy())

    return turns



def main():
    parser = argparse.ArgumentParser(description="ASR + Speaker Separation using HF Transformers")
    parser.add_argument("audio", help="Path to .wav or .mp3 file")
    parser.add_argument("--asr_model", default="openai/whisper-small", help="Hugging Face ASR model")
    parser.add_argument("--speaker_model", default="pyannote/speaker-diarization", help="Hugging Face speaker diarization model")
    parser.add_argument("--hf_token", default=os.getenv("HUGGINGFACE_TOKEN"), help="Hugging Face token for model access")
    args = parser.parse_args()

    if not args.hf_token:
        parser.error("Hugging Face token required via --hf_token or HUGGINGFACE_TOKEN env var")

    print("🔊 Running speaker diarization...")
    diarization_segments = diarize(args.audio, args.hf_token, args.speaker_model)

    print("📝 Transcribing audio...")
    asr_chunks = transcribe(args.audio, args.asr_model)

    print("🔗 Aligning ASR chunks to speakers...")
    transcript = assign_speakers(asr_chunks, diarization_segments)

    print("\n--- Transcript with Speaker Labels ---\n")
    for seg in transcript:
        print(f"[{seg['speaker']}] {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}")

if __name__ == "__main__":
    main()


