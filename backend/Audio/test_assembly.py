# test_transcription.py

from assembly_trans_diar import transcribe_audio_file

# Choose file
AUDIO_PATH = "/Users/natedamstra/Downloads/Mock Therapy Convo with Dimple.wav"
OUTPUT_JSON = "transcript_output.json"

result = transcribe_audio_file(AUDIO_PATH, save_json_path=OUTPUT_JSON)

print(f"Transcript ID: {result['id']}")
print("\n🎙️ Speaker-Diarized Transcript:")
for utt in result.get("utterances", []):
    print(f"[Speaker {utt['speaker']}] {utt['text']}")

print("\n📄 Summary:")
print(result.get("summary", "[No summary available]"))

print("\n💬 Sentiment Analysis:")
for s in result["sentiment_analysis_results"]:
    print(f"[{s['sentiment']} | {s['confidence']*100:.1f}%] {s['text']}")
