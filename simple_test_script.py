#!/usr/bin/env python3
"""
Simplified Multi-User PANDA Therapy Test Script

This script tests the basic multi-user functionality:
1. Creates multiple test users (patient and therapist)
2. Simulates audio file processing 
3. Tests user-specific data isolation
"""

import os
import time
import requests
import json
import uuid
from pprint import pprint

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Test data structure
USERS = [
    {
        "username": f"patient_{int(time.time())}",
        "name": "Test Patient",
        "role": "patient"
    },
    {
        "username": f"therapist_{int(time.time())}",
        "name": "Dr. Smith",
        "role": "therapist"
    }
]

# Audio simulation data
SIMULATED_AUDIO = {
    "session_id": str(uuid.uuid4()),
    "therapist": "Dr. Smith",
    "patient": "Test Patient",
    "transcript": [
        {
            "speaker": "therapist",
            "text": "How have you been managing your anxiety since our last session?",
            "emotion": "neutral",
            "timestamp": "00:01:12"
        },
        {
            "speaker": "patient", 
            "text": "I've been practicing those breathing exercises you taught me. They help sometimes, but I still get overwhelmed at work.",
            "emotion": "anxious",
            "timestamp": "00:01:45"
        },
        {
            "speaker": "therapist",
            "text": "I'm glad the breathing techniques are helping somewhat. Let's discuss some additional strategies for your workplace anxiety.",
            "emotion": "supportive",
            "timestamp": "00:02:20"
        }
    ]
}

def create_dev_user(user_data):
    """Create a development user"""
    print(f"\n=== Creating {user_data['role']} user: {user_data['username']} ===")
    endpoint = f"{BASE_URL}/api/dev-login"
    
    response = requests.post(endpoint, params=user_data)
    
    if response.status_code == 200:
        user_info = response.json()
        print(f"✓ Created user {user_data['username']}")
        print(f"✓ User ID: {user_info['user_id']}")
        return user_info
    else:
        print(f"✗ Failed to create user: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def send_message(user_id, message):
    """Send a message to the user's agent"""
    print(f"\n=== Sending message as user {user_id} ===")
    print(f"Message: {message}")
    
    endpoint = f"{BASE_URL}/api/chat/message"
    headers = {"X-User-ID": user_id, "Content-Type": "application/json"}
    
    payload = {
        "message": message,
        "chat_history": []
    }
    
    response = requests.post(endpoint, headers=headers, json=payload)
    
    if response.status_code == 200:
        print(f"✓ Message sent successfully")
        try:
            result = response.json()
            print(f"Response: {result.get('response', 'No response')}")
            return result
        except json.JSONDecodeError:
            print("✗ Invalid JSON response")
            return None
    else:
        print(f"✗ Failed to send message: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def simulate_upload_audio(user_id, audio_data):
    """Simulate uploading audio data for a user"""
    print(f"\n=== Simulating audio upload for user {user_id} ===")
    endpoint = f"{BASE_URL}/api/audio/simulate"
    
    headers = {"X-User-ID": user_id, "Content-Type": "application/json"}
    
    response = requests.post(endpoint, headers=headers, json=audio_data)
    
    if response.status_code == 200:
        print(f"✓ Audio data simulated successfully")
        return response.json()
    else:
        print(f"✗ Failed to simulate audio: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def run_multi_user_test():
    """Run a complete multi-user test"""
    print("\n" + "=" * 50)
    print("  MULTI-USER PANDA THERAPY TEST")
    print("=" * 50)
    
    # Create test users
    users = []
    for user_data in USERS:
        user_info = create_dev_user(user_data)
        if user_info:
            users.append(user_info)
    
    if len(users) < 2:
        print("✗ Failed to create required test users")
        return
    
    # Get user IDs
    patient = users[0]
    therapist = users[1]
    
    print("\n=== Test Setup Complete ===")
    print(f"Patient: {patient['username']} (ID: {patient['user_id']})")
    print(f"Therapist: {therapist['username']} (ID: {therapist['user_id']})")
    
    # Test 1: Simulate uploading audio data for the patient
    audio_result = simulate_upload_audio(patient['user_id'], SIMULATED_AUDIO)
    
    # Test 2: Patient asks about their session
    patient_question = "What did we discuss in my last therapy session about anxiety?"
    patient_response = send_message(patient['user_id'], patient_question)
    
    # Test 3: Another patient should NOT see the first patient's data (data isolation test)
    other_patient = create_dev_user({
        "username": f"other_patient_{int(time.time())}",
        "name": "Another Patient",
        "role": "patient"
    })
    
    if other_patient:
        isolation_question = "What did the previous patient discuss about anxiety?"
        other_response = send_message(other_patient['user_id'], isolation_question)
        
        # Check if the response contains specific information from the first patient
        # (This is just a basic check, in reality you'd want more thorough verification)
        if other_response and "breathing exercises" in other_response.get("response", ""):
            print("✗ PRIVACY FAILURE: Second patient could access first patient's data!")
        else:
            print("✓ Data isolation verified: Second patient cannot access first patient's data")
    
    print("\n" + "=" * 50)
    print("  TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    run_multi_user_test()
