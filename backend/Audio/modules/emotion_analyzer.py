"""
Emotion analysis module for therapy sessions.
Analyzes both audio and text for emotional content.
"""
import os
import json
from typing import Dict, List, Any, Optional
import datetime
from transformers import pipeline
from pydub import AudioSegment

class EmotionAnalyzer:
    """Analyzes emotions in both audio and text."""
    
    def __init__(self):
        """Initialize the emotion analysis models."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.audio_emotion_model = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        self.text_emotion_model = "j-hartmann/emotion-english-distilroberta-base"
        self._audio_classifier = None
        self._text_classifier = None
    
    @property
    def audio_classifier(self):
        """Lazy loading of audio emotion classifier."""
        if self._audio_classifier is None:
            self._audio_classifier = pipeline(
                "audio-classification", 
                model=self.audio_emotion_model, 
                top_k=3
            )
        return self._audio_classifier
    
    @property
    def text_classifier(self):
        """Lazy loading of text emotion classifier."""
        if self._text_classifier is None:
            self._text_classifier = pipeline(
                "text-classification", 
                model=self.text_emotion_model
            )
        return self._text_classifier
    
    def analyze_emotions(
        self, 
        audio_file_path: str, 
        transcript_path: str,
        output_dir: str = "../data/emotion_analysis"
    ) -> Dict[str, Any]:
        """
        Analyze emotions in audio and text from a therapy session.
        
        Args:
            audio_file_path: Path to the audio file
            transcript_path: Path to the transcript JSON with speaker assignments
            output_dir: Directory to save the emotion analysis results
            
        Returns:
            Dictionary with emotion analysis results
        """
        # Load the transcript
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
        
        # Load the audio file
        audio = AudioSegment.from_wav(audio_file_path)
        
        # Analyze each utterance
        results = []
        for utterance in transcript_data.get("utterances", []):
            try:
                start_ms = utterance.get("start", 0)
                end_ms = utterance.get("end", 0)
                speaker = utterance.get("speaker", "Unknown")
                text = utterance.get("text", "")
                
                if not text or end_ms <= start_ms:
                    continue
                
                # Extract audio segment for the utterance
                segment = audio[start_ms:end_ms]
                temp_filename = "temp_segment.wav"
                segment.export(temp_filename, format="wav")
                
                # Analyze audio emotion
                audio_prediction = self.audio_classifier(temp_filename)[0]
                
                # Analyze text emotion
                text_prediction = self.text_classifier(text)[0]
                
                # Create result entry
                result = {
                    "speaker": speaker,
                    "text": text,
                    "start": start_ms,
                    "end": end_ms,
                    "predicted_wav_emotion": audio_prediction["label"],
                    "confidence": audio_prediction["score"],
                    "predicted_text_emotion": text_prediction["label"],
                    "text_confidence": text_prediction["score"]
                }
                
                results.append(result)
                
                # Clean up temporary file
                os.remove(temp_filename)
                
            except Exception as e:
                print(f"Error processing utterance: {e}")
        
        # Create the complete analysis result
        analysis_result = {
            "audio_file": audio_file_path,
            "transcript_file": transcript_path,
            "emotion_analysis": results,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename based on input audio file name
            base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
            output_filename = f"{base_name}_emotions.json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            
            print(f"Emotion analysis results saved to: {os.path.abspath(output_path)}")
            analysis_result["output_path"] = output_path
        
        return analysis_result

# Simplified function for backward compatibility
def analyze_audio_emotions(
    audio_file_path: str,
    transcript_path: str,
    output_dir: str = "../data/emotion_analysis"
) -> str:
    """
    Analyze emotions in a therapy session and save results to JSON.
    
    Args:
        audio_file_path: Path to the audio file
        transcript_path: Path to the transcript JSON file with speaker labels
        output_dir: Output directory for the emotion analysis JSON
        
    Returns:
        Path to the output JSON file with emotion analysis results
    """
    analyzer = EmotionAnalyzer()
    result = analyzer.analyze_emotions(audio_file_path, transcript_path, output_dir)
    return result.get("output_path", "")

# Example usage
if __name__ == "__main__":
    # Test emotion analysis with sample files
    audio_path = "/path/to/audio/file.wav"
    transcript_path = "/path/to/transcript/file.json"
    
    analyzer = EmotionAnalyzer()
    results = analyzer.analyze_emotions(audio_path, transcript_path)
    
    # Print sample of results
    print(f"Analyzed {len(results.get('emotion_analysis', []))} utterances")
    if results.get("emotion_analysis"):
        sample = results["emotion_analysis"][0]
        print(f"Sample: {sample['speaker']} - '{sample['text']}'")
        print(f"Audio emotion: {sample['predicted_wav_emotion']} ({sample['confidence']:.2f})")
        print(f"Text emotion: {sample['predicted_text_emotion']} ({sample['text_confidence']:.2f})")
