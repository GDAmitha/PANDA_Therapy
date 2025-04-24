import os, json
from dotenv import load_dotenv
from audio_pipeline import process_audio_session

if __name__ == "__main__":
    # 1. Load your .env so HF and OpenAI tokens are available
    load_dotenv()

    audio_path = "Downloads/Mock Therapy Convo with Dimple.wav"
    
    result = process_audio_session(audio_path)
    print(json.dump(result, indent=2))