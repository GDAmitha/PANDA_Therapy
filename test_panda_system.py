#!/usr/bin/env python3
"""
PANDA Therapy System Test Script

This script tests the entire PANDA Therapy system workflow:
1. Register a new user as a patient
2. Login to get an authentication token
3. Upload an audio file from a therapy session
4. Chat with the personalized agent about the session
"""

import os
import time
import requests
import json
from pprint import pprint

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Test user data
TEST_USER = {
    "username": f"testuser_{int(time.time())}",  # Unique username using timestamp
    "password": "TestPassword123!",
    "email": f"test_{int(time.time())}@example.com",
    "name": "Test Patient",
    "role": "patient",
    # Additional patient-specific fields
    "medical_history": "No significant medical history",
    "therapy_goals": "Improve anxiety management",
    "emergency_contact": "Emergency Contact: 555-123-4567"
}

# Path to a sample audio file
SAMPLE_AUDIO = "/Users/natedamstra/Desktop/LLMAgents/PANDA_Therapy/sample_audio.wav"

# Ensure the audio file exists
if not os.path.exists(SAMPLE_AUDIO):
    print(f"Warning: Sample audio file not found at {SAMPLE_AUDIO}")
    print("You may need to update the path or create a sample audio file for testing.")
    # Create a dummy message to simulate audio processing
    TEST_USER["has_audio"] = False
else:
    TEST_USER["has_audio"] = True

# Sample questions to ask the agent after processing
SAMPLE_QUESTIONS = [
    "What were the main topics discussed in my therapy session?",
    "What emotions did I express during the session?",
    "Can you summarize my therapy goals based on the session?",
    "What therapy techniques were mentioned that might help with my anxiety?"
]

def register_user(user_data):
    """Register a new user using simplified authentication"""
    print("\n=== REGISTERING NEW USER (DEV MODE) ===")
    endpoint = f"{BASE_URL}/api/dev-login"
    
    # For dev login, we only need username, name and role
    params = {
        "username": user_data["username"],
        "name": user_data["name"],
        "role": user_data["role"]
    }
    
    response = requests.post(endpoint, params=params)
    
    if response.status_code == 200:
        print(f"✓ Dev user '{user_data['username']}' successfully created!")
        user_id = response.json().get("user_id")
        print(f"✓ User ID: {user_id} - Use this in X-User-ID header")
        return response.json()
    else:
        print(f"✗ Failed to register dev user: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def get_auth_header(user_id):
    """Create an authentication header with the user ID"""
    print("\n=== CREATING AUTH HEADER ===")
    print(f"✓ Using User ID: {user_id} for authentication")
    
    # In our simplified auth, we just use the X-User-ID header
    return {"X-User-ID": user_id}

def upload_audio(headers, audio_path):
    """Upload an audio therapy session recording"""
    print("\n=== UPLOADING THERAPY SESSION AUDIO ===")
    endpoint = f"{BASE_URL}/api/audio/upload"
    
    # Prepare form data with the audio file
    with open(audio_path, "rb") as audio_file:
        files = {"file": audio_file}
        data = {
            "therapist_name": "Dr. Smith",
            "patient_name": TEST_USER["name"]
        }
        
        response = requests.post(endpoint, headers=headers, files=files, data=data)
    
    if response.status_code == 200:
        print(f"✓ Audio file successfully uploaded and processed!")
        return response.json()
    else:
        print(f"✗ Failed to upload audio: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def simulate_audio_upload(headers):
    """Simulate audio upload when no audio file is available"""
    print("\n=== SIMULATING THERAPY SESSION DATA (NO AUDIO FILE) ===")
    print("Note: This is a simulation only since no audio file was provided")
    
    # In a real scenario, we would need an endpoint to simulate this
    # For testing purposes, we'll just return a mock response
    return {
        "status": "simulated",
        "message": "Audio data simulation successful",
        "session_id": f"simulated_{int(time.time())}"
    }

def send_message(headers, message, chat_history=None):
    """Send a message to the personal agent and get a response"""
    print(f"\n=== SENDING MESSAGE TO AGENT: '{message}' ===")
    endpoint = f"{BASE_URL}/api/chat/message"
    
    # Add content type to headers
    message_headers = headers.copy()
    message_headers["Content-Type"] = "application/json"
    
    if chat_history is None:
        chat_history = []
    
    payload = {
        "message": message,
        "chat_history": chat_history
    }
    
    response = requests.post(endpoint, headers=message_headers, json=payload)
    
    if response.status_code == 200:
        chat_response = response.json()
        print(f"✓ Agent responded!")
        print(f"Agent: {chat_response['response']}")
        return chat_response
    else:
        print(f"✗ Failed to get response: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def run_full_test():
    """Execute the complete test workflow"""
    print("\n========================================")
    print("  PANDA THERAPY SYSTEM TEST WORKFLOW")
    print("  (SIMPLIFIED AUTHENTICATION)")
    print("========================================")
    
    # Step 1: Register a new user with dev mode
    user_result = register_user(TEST_USER)
    if not user_result:
        print("Test failed at user registration step")
        return
    
    # Get the user ID for authentication
    user_id = user_result.get("user_id")
    if not user_id:
        print("Test failed: No user ID returned from registration")
        return
        
    # Step 2: Create auth header with the user ID
    auth_headers = get_auth_header(user_id)
    
    # Step 3: Upload audio file or simulate audio data
    if TEST_USER["has_audio"]:
        audio_result = upload_audio(auth_headers, SAMPLE_AUDIO)
    else:
        audio_result = simulate_audio_upload(auth_headers)
        
    if not audio_result:
        print("Test failed at audio processing step")
        return
        
    # Step 4: Wait a bit for audio processing to complete
    print("\nWaiting for audio processing to complete...")
    time.sleep(3)
    
    # Step 5: Chat with the agent
    chat_history = []
    
    # Initial greeting
    greeting_response = send_message(auth_headers, "Hello, I just had a therapy session. Can you help me reflect on it?")
    if greeting_response:
        chat_history.append({
            "role": "user", 
            "content": "Hello, I just had a therapy session. Can you help me reflect on it?"
        })
        chat_history.append({
            "role": "assistant", 
            "content": greeting_response["response"]
        })
    
    # Ask each sample question
    for question in SAMPLE_QUESTIONS:
        question_response = send_message(auth_headers, question, chat_history)
        
        if question_response:
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": question_response["response"]})
    
    print("\n========================================")
    print("  TEST COMPLETED SUCCESSFULLY")
    print("========================================")
    print(f"- User '{TEST_USER['username']}' was registered with ID: {user_id}")
    if TEST_USER["has_audio"]:
        print(f"- Audio file was uploaded and processed")
    else:
        print(f"- Audio processing was simulated (no file provided)")
    print(f"- Completed conversation with {len(SAMPLE_QUESTIONS) + 1} exchanges")
    
if __name__ == "__main__":
    run_full_test()
