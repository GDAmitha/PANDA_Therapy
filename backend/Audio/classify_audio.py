# therapy_graph.py
import getpass
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

# --- Step 1: Transcription Node (AssemblyAI) ---
from assembly_trans_diar import transcribe_audio_file

def transcribe_node(state):
    audio_path = state["audio_path"]
    result = transcribe_audio_file(audio_path)
    state["utterances"] = result["utterances"]
    state["raw_transcript"] = result["text"]
    return state

# --- Step 2: Speaker Classification Node ---
from nodes.speaker_classifier_node import classify_speaker_roles_node

# --- Step 3: Emotion Analysis Node using GPT ---
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

emotion_prompt = ChatPromptTemplate.from_template("""
You are an expert at understanding emotional states from therapy transcripts.
Below are client utterances. For each one, classify the dominant emotion.

{utterances}

Return a JSON list like:
[
  {{"text": "...", "emotion": "anxious"}},
  {{"text": "...", "emotion": "hopeful"}}
]
""")

def analyze_emotion_node(state):
    client_utts = [utt["text"] for utt in state["utterances"] if utt.get("role") == "client"]
    chunked = "\n".join(f"- {utt}" for utt in client_utts[:10])  # limit to 10 for context

    chat = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = emotion_prompt.format_messages(utterances=chunked)
    result = chat(prompt)
    state["emotion_analysis"] = result.content.strip()
    return state

# --- Graph Construction ---
graph = StateGraph()

graph.add_node("transcribe", RunnableLambda(transcribe_node))
graph.add_node("classify_roles", RunnableLambda(classify_speaker_roles_node))
graph.add_node("analyze_emotion", RunnableLambda(analyze_emotion_node))

# Define edges (flow)
graph.set_entry_point("transcribe")
graph.add_edge("transcribe", "classify_roles")
graph.add_edge("classify_roles", "analyze_emotion")

# Compile
app = graph.compile()

if __name__ == "__main__":
    # Run full pipeline on a test file
    initial_state = {"audio_path": "/path/to/audio.wav"}
    final_state = app.invoke(initial_state)

    print("\n\n🎙️ Emotion Analysis Result:\n")
    print(final_state["emotion_analysis"])
