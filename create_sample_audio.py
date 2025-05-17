#!/usr/bin/env python3
"""
Create a sample audio file for PANDA Therapy testing
"""

import numpy as np
from scipy.io.wavfile import write
import os

def create_sample_audio():
    """Create a simple sine wave audio file for testing"""
    print("Creating sample audio file...")
    
    # Parameters
    sample_rate = 44100  # Sample rate in Hz
    duration = 10  # Duration in seconds
    frequency = 440  # Frequency in Hz (A4 note)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate sine wave
    note = np.sin(2 * np.pi * frequency * t)
    
    # Add some variation to simulate speech
    note = note * np.sin(2 * np.pi * 0.5 * t)
    
    # Normalize
    audio = note * 0.3
    audio = np.int16(audio * 32767)
    
    # Write to file
    output_path = os.path.join(os.path.dirname(__file__), "sample_audio.wav")
    write(output_path, sample_rate, audio)
    
    print(f"Sample audio created at: {output_path}")
    return output_path

if __name__ == "__main__":
    create_sample_audio()
