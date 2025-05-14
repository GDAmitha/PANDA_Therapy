import os
import requests
import time
from dotenv import load_dotenv

base_url = "https://api.assemblyai.com"

load_dotenv()
AAPI_KEY = os.getenv("ASSEMBLY_TOKEN")

headers = {
    "authorization": AAPI_KEY
}

# URL of the file to transcribe
FILE_PATH = "/Users/natedamstra/Downloads/Mock Therapy Convo with Dimple.wav"

with open(FILE_PATH, "rb") as f:
    response = requests.post(base_url + "/v2/upload", headers=headers, data=f)
    if response.status_code != 200:
        print(f"Error: {response.status_code}, Response: {response.text}")
        response.raise_for_status()
    upload_json = response.json()
    FILE_URL = upload_json["upload_url"]

# You can set additional parameters for the transcription
config = {
  "audio_url": FILE_URL,
  "speech_model":"slam-1",
  "summarization": True,
  "content_safety": True,
  "iab_categories": True,
  "sentiment_analysis": True,
  "speaker_labels": True,
  "language_code": "en_us"
}



url = base_url + "/v2/transcript"
response = requests.post(url, json=config, headers=headers)

transcript_id = response.json()['id']
polling_endpoint = base_url + "/v2/transcript/" + transcript_id

while True:
  transcription_result = requests.get(polling_endpoint, headers=headers).json()
  transcription_text = transcription_result['text']
  ##summary = transcription_result['summary']

  if transcription_result['status'] == 'completed':
    print(f"Transcript ID: ", transcript_id)
    print("\n🎙️ Speaker-Diarized Transcript:")
    for utt in transcription_result.get('utterances', []):
        print(f"[Speaker {utt['speaker']}] {utt['text']}")

    print("\n🎙️ Summarized Transcript:")
    print(transcription_result['summary'])
    print("\n Content Safety Data:")
    for result in transcription_result['content_safety_labels']['results']:
        print(result['text'])
        print(f"Timestamp: {result['timestamp']['start']} - {result['timestamp']['end']}")
        # Get category, confidence, and severity.
        for label in result['labels']:
            print(f"{label['label']} - {label['confidence']} - {label['severity']}")  # content safety category
    # Get the confidence of the most common labels in relation to the entire audio file.
    for label, confidence in transcription_result['content_safety_labels']['summary'].items():
        print(f"{confidence * 100}% confident that the audio contains {label}")
    # Get the overall severity of the most common labels in relation to the entire audio file.
    for label, severity_confidence in transcription_result['content_safety_labels']['severity_score_summary'].items():
        print(f"{severity_confidence['low'] * 100}% confident that the audio contains low-severity {label}")
        print(f"{severity_confidence['medium'] * 100}% confident that the audio contains medium-severity {label}")
        print(f"{severity_confidence['high'] * 100}% confident that the audio contains high-severity {label}")
    
    print("Sentiment Analysis:")
    for sentiment_result in transcription_result['sentiment_analysis_results']:
        print(sentiment_result['text'])
        print(sentiment_result['sentiment'])  # POSITIVE, NEUTRAL, or NEGATIVE
        print(sentiment_result['confidence'])
        print(f"Timestamp: {sentiment_result['start']} - {sentiment_result['end']}")
    break


  elif transcription_result['status'] == 'error':
    raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

  else:
    time.sleep(3)

  
