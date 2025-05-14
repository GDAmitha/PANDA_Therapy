import os
import json
from dotenv import load_dotenv
from Audio2 import transcribe_and_segment

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    audio_file = "/Users/natedamstra/Downloads/Mock Therapy Convo with Dimple.wav"
    print("Running full ASR + diarization pipeline…")
    turns = transcribe_and_segment(audio_file)
    print(f"Generated {len(turns)} speaker turns.")

    # preview first few turns
    for t in turns[:5]:
        print(f"[{t['speaker']}] {t['start']:.2f}s–{t['end']:.2f}s: {t['text']}")

    # save JSON
    outpath = "test_transcript.json"
    with open(outpath, "w") as f:
        json.dump(turns, f, indent=2)
    print(f"Saved transcript to {outpath}")

if __name__ == "__main__":
    main()
