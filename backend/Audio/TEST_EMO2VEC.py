import os
from dotenv import load_dotenv
import torch
import soundfile as sf
import numpy as np
import librosa
from transformers import (
    pipeline,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
# Use ehcalabres model for audio emotions




# inference_pipeline = pipeline(
#     "",
#     model="emotion2vec/emotion2vec_plus_base")

# rec_result = inference_pipeline('https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav', granularity="utterance", extract_embedding=False)
# print(rec_result)


emotion_pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
result = emotion_pipe("I'm feeling overwhelmed lately.")
print(result)