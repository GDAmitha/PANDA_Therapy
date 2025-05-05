## DEPRECIATED ###




# import os, json
# from dotenv import load_dotenv
# from audio_pipeline import process_audio_session

# if __name__ == "__main__":
#     # 1. Load your .env so HF and OpenAI tokens are available
#     load_dotenv()

#     audio_path = "Downloads/Mock Therapy Convo with Dimple.wav"
    
#     result = process_audio_session(audio_path)
#     print(json.dump(result, indent=2))

# test_hf_asr_diarization.py
import os
from dotenv import load_dotenv

# 1) bring in your .env
load_dotenv()  
HF_TOKEN = os.getenv("HF_TOKEN")

# 2) import the functions you wrote
from Audio_classification import diarize, transcribe, assign_speakers, merge_speaker_turns

def main():
    audio_file = "/Users/natedamstra/Downloads/Mock Therapy Convo with Dimple.wav"  # path to your test audio

    # 3) run diarization
    print("Running diarization…")
    diarization = diarize(audio_file, HF_TOKEN, speaker_model="pyannote/speaker-diarization")
    
    # 4) run ASR
    print("Running ASR…")
    #asr_chunks = transcribe(audio_file, asr_model="openai/whisper-small")
    transcipt = transcribe(audio_file, diarization, asr_model="openai/whisper-small")

    # 6) inspect / assert some basics
    assert isinstance(transcript, list), "Expected a list of segments"
    print(f"Got {len(transcript)} segments.\n")
    
    # 7) print out a few lines
    for seg in transcript[:5]:
        print(f"[{seg['speaker']}] {seg['start']:.2f}s–{seg['end']:.2f}s: {seg['text']!r}")

    # 8) (optionally) write JSON
    import json
    with open("test_transcript.json","w") as f:
        json.dump(transcript, f, indent=2)
    print("\nSaved full JSON to test_transcript.json")

if __name__ == "__main__":
    main()


    # # 5) align
    # print("Aligning segments…")
    # transcript = assign_speakers(asr_chunks, diarization)

    # # 5.2) merge word segments into speaker sentences
    # print("Merging aligned word segments into sentences")
    # transcript = merge_speaker_turns(transcript, 1.0)