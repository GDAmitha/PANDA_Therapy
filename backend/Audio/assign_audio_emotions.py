
from transformers import pipeline
from pydub import AudioSegment
import json
import os
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def assign_audio_emotions(audio_file_path: str, labeled_transcript_path: str):
    """
    Analyze audio segments for emotion and save results to a JSON file.
    
    Args:
        audio_file_path (str): Path to the audio file
        labeled_transcript_path (str): Path to the transcript JSON file with speaker labels
        
    Returns:
        str: Path to the output JSON file with emotion analysis results
    """
    emotion_classifier = pipeline("audio-classification", 
                                 model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", 
                                 top_k=3)
    text_classifier = pipeline("text-classification", 
                               model="j-hartmann/emotion-english-distilroberta-base")
    # Load the transcript JSON
    with open(labeled_transcript_path, 'r') as f:
        transcript_data = json.load(f)

    # Load the full audio file
    audio = AudioSegment.from_wav(audio_file_path)

    results = []

    for utterance in transcript_data["utterances"]:
        start_ms = utterance["start"]
        end_ms = utterance["end"]
        speaker = utterance["speaker"]
        text = utterance["text"]

        # Extract the audio segment for the utterance
        segment = audio[start_ms:end_ms]

        # Export the segment to a temporary file
        temp_filename = "temp_segment.wav"
        segment.export(temp_filename, format="wav")

        # Use the classifier on the audio segment
        prediction = emotion_classifier(temp_filename)[0]  # Get the top prediction
        print(prediction)
        
        # Use the text classifier on the text
        text_prediction = text_classifier(text)[0]
        print(text_prediction)
        # Append the result
        results.append({
            "speaker": speaker,
            "text": text,
            "start": start_ms,
            "end": end_ms,
            "predicted_wav_emotion": prediction["label"],
            "confidence": prediction["score"],
            "predicted_text_emotion": text_prediction["label"],
            "text_confidence": text_prediction["score"]
        })

        # Remove the temporary file
        os.remove(temp_filename)


    
    
    # Create output directory if it doesn't exist
    output_dir = "audio_emo_transcript"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on input audio file name
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_filename = f"{base_name}_emotions.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save results to JSON file
    with open(output_path, 'w') as f:
        json.dump({
            'audio_file': audio_file_path,
            'transcript_file': labeled_transcript_path,
            'emotion_analysis': results
        }, f, indent=2)
    
    print(f"Emotion analysis results saved to: {os.path.abspath(output_path)}")
    return output_path


def main():
    assign_audio_emotions("/Users/natedamstra/Downloads/Mock Therapy Convo with Dimple.wav", "speaker_assign_transcript/Mock Therapy Convo with Dimple_transcript_assigned.json")


if __name__ == "__main__":
    main()