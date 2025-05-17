"""
PANDA Therapy Development UI

A simple Streamlit app to interact with the PANDA Therapy backend API.
"""

import streamlit as st
import requests
import json
import os
import time
from datetime import datetime
import pandas as pd
import uuid

# API Configuration
API_URL = "http://localhost:8000"

# Page setup
st.set_page_config(
    page_title="PANDA Therapy Dev UI", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent_id' not in st.session_state:
    st.session_state.agent_id = None
if 'simulated_audio' not in st.session_state:
    st.session_state.simulated_audio = []

# Authentication Functions
def dev_login(username, name, role="patient"):
    """Login or register a development user"""
    response = requests.post(
        f"{API_URL}/api/dev-login",
        params={"username": username, "name": name, "role": role}
    )
    
    if response.status_code == 200:
        data = response.json()
        st.session_state.user_id = data['user_id']
        st.session_state.username = data['username']
        st.session_state.role = data['role']
        st.session_state.agent_id = data.get('agent_id')
        return True
    else:
        st.error(f"Login failed: {response.text}")
        return False

def logout():
    """Clear user session"""
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.chat_history = []
    st.session_state.agent_id = None

# API Interaction Functions
def send_message(message):
    """Send message to the agent and get response"""
    headers = {
        "X-User-ID": st.session_state.user_id,
        "Content-Type": "application/json"
    }
    
    payload = {
        "message": message,
        "chat_history": st.session_state.chat_history
    }
    
    response = requests.post(
        f"{API_URL}/api/chat/message",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json().get("response", "No response")
    else:
        st.error(f"Error sending message: {response.text}")
        return None

def simulate_audio_session(transcript_data):
    """Simulate a therapy audio session"""
    headers = {
        "X-User-ID": st.session_state.user_id,
        "Content-Type": "application/json"
    }
    
    session_id = str(uuid.uuid4())
    
    # Create structured transcript data
    audio_data = {
        "session_id": session_id,
        "therapist": "Dr. Smith",
        "patient": "Test Patient",
        "transcript": transcript_data
    }
    
    response = requests.post(
        f"{API_URL}/api/audio/simulate",
        headers=headers,
        json=audio_data
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error simulating audio: {response.text}")
        return None

def upload_audio_file(audio_file, therapist_name, patient_name):
    """Upload an audio file to the API"""
    headers = {
        "X-User-ID": st.session_state.user_id
    }
    
    files = {"file": audio_file}
    data = {
        "therapist_name": therapist_name,
        "patient_name": patient_name
    }
    
    response = requests.post(
        f"{API_URL}/api/audio/upload",
        headers=headers,
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error uploading audio: {response.text}")
        return None

# UI Components
def login_page():
    """Render login/registration page"""
    st.title("PANDA Therapy Development UI")
    
    with st.container():
        st.subheader("User Login / Registration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username")
            name = st.text_input("Full Name")
            
        with col2:
            role = st.selectbox(
                "Role",
                options=["patient", "therapist", "admin"],
                index=0
            )
        
        if st.button("Login / Register"):
            if username and name:
                if dev_login(username, name, role):
                    st.success(f"Logged in as {name} ({role})")
                    st.rerun()
            else:
                st.error("Please enter both username and name")

def main_app():
    """Render main application after login"""
    # Sidebar with user info and logout
    with st.sidebar:
        st.subheader("User Information")
        st.write(f"Username: {st.session_state.username}")
        st.write(f"Role: {st.session_state.role}")
        st.write(f"User ID: {st.session_state.user_id}")
        
        if st.session_state.agent_id:
            st.write(f"Agent ID: {st.session_state.agent_id}")
        
        if st.button("Logout"):
            logout()
            st.rerun()
        
        st.divider()
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio("Go to", ["Chat", "Audio Upload", "Simulate Session"])
    
    # Main content area
    if page == "Chat":
        chat_page()
    elif page == "Audio Upload":
        audio_upload_page()
    elif page == "Simulate Session":
        simulate_session_page()

def chat_page():
    """Chat interface with the agent"""
    st.title("Chat with Your Therapy Agent")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You**: {message['content']}")
            else:
                st.markdown(f"**Agent**: {message['content']}")
    
    # Message input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your message:", height=100)
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Get agent response
            response = send_message(user_input)
            
            if response:
                # Add agent response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
            
            # Force refresh to show new messages
            st.rerun()

def audio_upload_page():
    """Interface for uploading audio files"""
    st.title("Upload Therapy Session Audio")
    
    with st.form(key="audio_upload_form"):
        audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            therapist_name = st.text_input("Therapist Name", value="Dr. Smith")
        
        with col2:
            patient_name = st.text_input("Patient Name", value="Test Patient")
        
        submit_button = st.form_submit_button("Upload")
        
        if submit_button and audio_file:
            result = upload_audio_file(audio_file, therapist_name, patient_name)
            
            if result:
                st.success(f"Audio uploaded successfully! Session ID: {result.get('session_id')}")
                st.json(result)

def simulate_session_page():
    """Interface for simulating a therapy session"""
    st.title("Simulate Therapy Session")
    
    # Transcript builder
    st.subheader("Build Transcript")
    
    with st.form(key="transcript_form"):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            text = st.text_area("Statement text")
        
        with col2:
            speaker = st.selectbox("Speaker", ["therapist", "patient"])
            emotion = st.selectbox("Emotion", ["neutral", "positive", "negative", "anxious", "supportive", "frustrated"])
            
        with col3:
            timestamp = st.text_input("Timestamp", value="00:00:00")
            add_button = st.form_submit_button("Add Statement")
            
        if add_button and text:
            # Add statement to simulated audio
            st.session_state.simulated_audio.append({
                "speaker": speaker,
                "text": text,
                "emotion": emotion,
                "timestamp": timestamp
            })
    
    # Display current transcript
    if st.session_state.simulated_audio:
        st.subheader("Current Transcript")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(st.session_state.simulated_audio)
        st.dataframe(df, use_container_width=True)
        
        # Buttons to manage transcript
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Transcript"):
                st.session_state.simulated_audio = []
                st.rerun()
        
        with col2:
            if st.button("Submit Transcript"):
                if len(st.session_state.simulated_audio) > 0:
                    result = simulate_audio_session(st.session_state.simulated_audio)
                    
                    if result:
                        st.success("Therapy session simulated successfully!")
                        st.json(result)
                        # Clear transcript after successful submission
                        st.session_state.simulated_audio = []
                        st.rerun()
                else:
                    st.error("Please add at least one statement to the transcript")

# Main app flow
def main():
    if st.session_state.user_id:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()
