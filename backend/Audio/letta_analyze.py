from letta_client import Letta
import json
import os

client = Letta(token=os.getenv("LETTA_API_KEY"))

agent_analyze = client.agents.retrieve(agent_id="agent-e6f7f7c3-6063-4929-9e20-62716c7999d1")

def analyze_therapy_session(transcript_path: str) -> str:
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    client.agents.messages.create(
        agent_id=agent_analyze.id,
        messages=[{
            "role": "system", 
            "content": "You are a therapist analyzing a therapy session. you will be given a speakers turn text and emotional analysis in many prompts in order. follow your system instructions and analyze the session. I'll tell you when we are done for this session."
            }]
    )
    for utterance in transcript_data["emotion_analysis"]:
        response = client.agents.messages.create(
            agent_id=agent_analyze.id,
            messages=[{
                "role": "system", 
                "content": f"Speaker: {utterance["speaker"]}\n Text: {utterance["text"]}\n Predicted WAV Emotion: {utterance["predicted_wav_emotion"]}\n Predicted Text Emotion: {utterance["predicted_text_emotion"]}\n Confidence for WAV Emotion: {utterance["confidence"]}\n Confidence for Text Emotion: {utterance["text_confidence"]}\n"
            }]
        )
        for message in response.messages:
            print_message(message)
    client.agents.messages.create(
        agent_id=agent_analyze.id,
        messages=[{
            "role": "system", 
            "content": "Good job! We are done analyzing this therapy session. Now you can answer any questions from the user and use this information to assist them in there therapuetic needs. Also be ready to summarize there session for them and provide them with key insights and be able to quote directly from the therapist!"
        }]
    )
    
def print_message(message):
    if message.message_type == "reasoning_message":
        print("Reasoning:", message.reasoning)
    elif message.message_type == "assistant_message":
        print("Agent:", message.content)
    elif message.message_type == "tool_call_message":
        print("Tool Call:", message.tool_call.name)
        print("Arguments:", message.tool_call.arguments)
    elif message.message_type == "tool_return_message":
        print("Tool Return:", message.tool_return)
    elif message.message_type == "user_message":
        print("User Message:", message.content)

analyze_therapy_session("audio_emo_transcript/Mock Therapy Convo with Dimple_emotions.json")
