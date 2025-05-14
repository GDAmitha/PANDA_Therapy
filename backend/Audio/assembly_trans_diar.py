# assembly_trans_diar.py

import os
import requests
import time
import json
from dotenv import load_dotenv

load_dotenv()
AAPI_KEY = os.getenv("ASSEMBLY_TOKEN")
BASE_URL = "https://api.assemblyai.com"

HEADERS = {"authorization": AAPI_KEY}


def transcribe_audio_file(file_path: str, save_json_path: str = None) -> dict:
    """
    Uploads, transcribes, and retrieves enriched transcript from AssemblyAI.
    Saves full JSON to `save_json_path` if provided.
    """
    # Upload file
    with open(file_path, "rb") as f:
        res = requests.post(f"{BASE_URL}/v2/upload", headers=HEADERS, data=f)
        if res.status_code != 200:
            raise RuntimeError(f"Upload failed: {res.status_code} - {res.text}")
        FILE_URL = res.json()["upload_url"]

    # Request transcription
    config = {
        "audio_url": FILE_URL,
        "speech_model": "slam-1",
        "summarization": True,
        "iab_categories": True,
        "sentiment_analysis": True,
        "speaker_labels": True,
        "language_code": "en_us",
    }

    res = requests.post(f"{BASE_URL}/v2/transcript", json=config, headers=HEADERS)
    if res.status_code != 200:
        raise RuntimeError(f"Transcription request failed: {res.status_code} - {res.text}")
    transcript_id = res.json()["id"]

    # Polling loop
    polling_endpoint = f"{BASE_URL}/v2/transcript/{transcript_id}"
    while True:
        res = requests.get(polling_endpoint, headers=HEADERS)
        data = res.json()
        if data["status"] == "completed":
            if save_json_path:
                with open(save_json_path, "w") as f:
                    json.dump(data, f, indent=2)
            return data
        elif data["status"] == "error":
            raise RuntimeError(f"Transcription failed: {data['error']}")
        time.sleep(3)

