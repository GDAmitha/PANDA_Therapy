from transformers import pipeline
from pydub import AudioSegment
import json
import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for transformer models
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_emotions_from_keywords(text: str) -> List[str]:
    """
    Extract emotions from text using simple keyword matching.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of detected emotions, sorted by confidence
    """
    if not text:
        return ["neutral"]
        
    text = text.lower()
    emotion_keywords = {
        "joy": ["happy", "joy", "excited", "glad", "pleased", "delighted", "cheerful", "smile"],
        "sadness": ["sad", "unhappy", "depressed", "down", "upset", "miserable", "sorrow", "grief", "cry"],
        "anger": ["angry", "mad", "furious", "rage", "annoyed", "irritated", "frustrated"],
        "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous", "panic"],
        "surprise": ["surprised", "shocked", "amazed", "astonished", "wow"],
        "disgust": ["disgusted", "grossed", "repulsed", "revolted", "yuck"],
        "neutral": ["okay", "fine", "alright", "normal", "so-so"]
    }
    
    # Count matches for each emotion
    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 1
        if score > 0:
            emotion_scores[emotion] = score
    
    # Default to neutral if no emotions detected
    if not emotion_scores:
        return ["neutral"]
    
    # Return emotions sorted by score
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    return [emotion for emotion, _ in sorted_emotions]

def assign_audio_emotions(audio_file_path: str, labeled_transcript_path: str) -> str:
    """
    Analyze audio segments for emotion and save results to a JSON file.
    
    Args:
        audio_file_path (str): Path to the audio file
        labeled_transcript_path (str): Path to the transcript JSON file with speaker labels
        
    Returns:
        str: Path to the output JSON file with emotion analysis results
    """
    logger.info(f"Starting emotion analysis for {labeled_transcript_path}")
    
    # Initialize results list
    results = []
    
    try:
        # Initialize emotion classifiers with error handling
        models_loaded = False
        emotion_classifier = None
        text_classifier = None
        
        try:
            emotion_classifier = pipeline("audio-classification", 
                                    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", 
                                    top_k=3)
            text_classifier = pipeline("text-classification", 
                                  model="j-hartmann/emotion-english-distilroberta-base")
            models_loaded = True
            logger.info("Emotion analysis models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load emotion models: {str(e)}")
        
        # Load the transcript JSON
        with open(labeled_transcript_path, 'r') as f:
            transcript_data = json.load(f)
        
        # Check for utterances key
        if "utterances" not in transcript_data:
            logger.warning("No 'utterances' key found in transcript data - creating empty list")
            transcript_data["utterances"] = []

        # Check if audio file exists
        audio = None
        if os.path.exists(audio_file_path):
            try:
                audio = AudioSegment.from_wav(audio_file_path)
                logger.info(f"Loaded audio file: {audio_file_path}")
            except Exception as e:
                logger.error(f"Failed to load audio file: {str(e)}")
        else:
            logger.error(f"Audio file not found: {audio_file_path}")
            # Continue with just text analysis

        # Process each utterance in the transcript
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
            for utterance in transcript_data.get("utterances", []):
                speaker = utterance.get("speaker", "")
                text = utterance.get("text", "")
                start_ms = int(utterance.get("start_time_ms", 0))
                end_ms = int(utterance.get("end_time_ms", 0))
                
                # Initialize emotion variables
                audio_emotions = []
                text_emotions = []
                keyword_emotions = []
                
                # Process audio if we have a valid segment
                if start_ms < end_ms and audio and models_loaded and emotion_classifier:
                    try:
                        # Extract segment and save to temp file
                        audio_segment = audio[start_ms:end_ms]
                        audio_segment.export(temp_path, format="wav")
                        
                        # Analyze audio for emotions
                        audio_results = emotion_classifier(temp_path)
                        
                        # Extract emotions with scores
                        for item in audio_results:
                            audio_emotions.append({
                                "label": item["label"],
                                "score": float(item["score"])
                            })
                        
                        logger.info(f"Audio emotions for segment {start_ms}-{end_ms}: {audio_emotions}")
                    except Exception as e:
                        logger.error(f"Error processing audio segment {start_ms}-{end_ms}: {str(e)}")
                
                # Process text for emotions
                if text and models_loaded and text_classifier:
                    try:
                        # Analyze text for emotions
                        text_results = text_classifier(text)
                        
                        # Extract emotions with scores
                        for item in text_results:
                            text_emotions.append({
                                "label": item["label"],
                                "score": float(item["score"])
                            })
                        
                        logger.info(f"Text emotions for '{text[:30]}...': {text_emotions}")
                    except Exception as e:
                        logger.error(f"Error processing text emotions: {str(e)}")
                
                # Use keyword-based emotion detection as a fallback
                try:
                    extracted_emotions = extract_emotions_from_keywords(text)
                    for emotion in extracted_emotions:
                        keyword_emotions.append({
                            "label": emotion,
                            "score": 0.75  # Default confidence for keyword emotions
                        })
                    logger.info(f"Keyword emotions: {keyword_emotions}")
                except Exception as e:
                    logger.error(f"Error extracting keyword emotions: {str(e)}")
                
                # Combine and score emotions from all sources
                combined_emotions = {}
                
                # Add audio emotions (highest weight)
                for emotion in audio_emotions:
                    label = emotion["label"]
                    score = emotion["score"] * 2.0  # Higher weight for audio
                    combined_emotions[label] = combined_emotions.get(label, 0) + score
                
                # Add text emotions (medium weight)
                for emotion in text_emotions:
                    label = emotion["label"]
                    score = emotion["score"] * 1.5  # Medium weight for text
                    combined_emotions[label] = combined_emotions.get(label, 0) + score
                
                # Add keyword emotions (lowest weight, but still valuable)
                for emotion in keyword_emotions:
                    label = emotion["label"]
                    score = emotion["score"] * 1.0  # Lower weight for keywords
                    combined_emotions[label] = combined_emotions.get(label, 0) + score
                
                # Sort emotions by score
                sorted_emotions = sorted(combined_emotions.items(), key=lambda x: x[1], reverse=True)
                
                # Calculate confidence based on original model scores
                wav_confidence = 0.7  # Default
                text_confidence = 0.7  # Default
                
                # Get actual confidence scores if available
                if audio_emotions and len(audio_emotions) > 0:
                    wav_confidence = audio_emotions[0]["score"]
                if text_emotions and len(text_emotions) > 0:
                    text_confidence = text_emotions[0]["score"]
                    
                # Get final emotion list 
                if sorted_emotions:
                    # Get up to top 2 emotions
                    emotions = [emotion for emotion, _ in sorted_emotions[:2]]
                    
                    # Use the highest confidence score from either model, with preference for audio
                    raw_confidence = max(wav_confidence, text_confidence * 0.9)  # Slight preference for audio
                    
                    # Boost confidence if multiple models agreed on the emotion
                    top_emotion = sorted_emotions[0][0]
                    agreement_boost = 0.0
                    
                    for emotion in audio_emotions:
                        if emotion["label"] == top_emotion:
                            agreement_boost += 0.1
                    
                    for emotion in text_emotions:
                        if emotion["label"] == top_emotion:
                            agreement_boost += 0.1
                            
                    # Calculate final confidence (with limits)
                    confidence = min(0.95, max(0.7, raw_confidence + agreement_boost))
                else:
                    # Default if no emotions detected
                    emotions = ["neutral"]
                    confidence = 0.5
                
                # Create the result object
                result = {
                    "speaker": speaker,
                    "text": text,
                    "emotions": emotions,
                    "confidence": float(confidence)
                }
                
                results.append(result)
                
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Error removing temp file: {str(e)}")

        # Create output directory in the Audio folder
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get Audio folder path
        output_dir = os.path.join(current_dir, "audio_emo_transcript")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        file_name = os.path.basename(labeled_transcript_path)
        base_name = os.path.splitext(file_name)[0]  # Remove extension
        output_path = os.path.join(output_dir, f"{base_name}_emotions.json")
        
        # Create the output data structure
        output_data = {
            "transcript": transcript_data.get("utterances", []),
            "emotion_analysis": results,
            "processed_at": Path(labeled_transcript_path).stat().st_mtime
        }
        
        # Save the results
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Emotion analysis saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}")
        
        # Create a minimal output with error information
        output_dir = os.path.join(os.path.dirname(os.path.abspath(labeled_transcript_path)), "audio_emo_transcript")
        os.makedirs(output_dir, exist_ok=True)
        
        file_name = os.path.basename(labeled_transcript_path)
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}_emotions.json")
        
        # Add error info to output
        error_output = {
            "error": str(e),
            "transcript": [],
            "emotion_analysis": []
        }
        
        # Try to load the transcript if possible
        try:
            with open(labeled_transcript_path, 'r') as f:
                transcript_data = json.load(f)
                error_output["transcript"] = transcript_data.get("utterances", [])
                
                # Add basic emotion data using keyword extraction
                for utterance in error_output["transcript"]:
                    text = utterance.get("text", "")
                    speaker = utterance.get("speaker", "Unknown")
                    emotions = extract_emotions_from_keywords(text)
                    
                    result = {
                        "speaker": speaker,
                        "text": text,
                        "predicted_wav_emotion": emotions[0] if emotions else "neutral",
                        "predicted_text_emotion": emotions[0] if emotions else "neutral",
                        "confidence": 0.5,
                        "text_confidence": 0.5
                    }
                    
                    error_output["emotion_analysis"].append(result)
        except Exception:
            pass  # Already in error handling, just continue
            
        # Save the fallback results
        with open(output_path, 'w') as f:
            json.dump(error_output, f, indent=2)
            
        logger.info(f"Fallback emotion analysis saved to {output_path}")
        return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        audio_file = sys.argv[1]
        transcript_file = sys.argv[2]
        
        output_path = assign_audio_emotions(audio_file, transcript_file)
        print(f"Emotion analysis completed and saved to {output_path}")
    else:
        print("Usage: python assign_audio_emotions.py <audio_file> <transcript_file>")
        sys.exit(1)