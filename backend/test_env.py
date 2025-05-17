"""
Test script to check environment variable loading
"""
import os
import dotenv
import sys

# Print Python version
print(f"Python version: {sys.version}")

# Path to .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
print(f"Looking for .env file at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")

# Try to load the .env file
if os.path.exists(env_path):
    dotenv.load_dotenv(dotenv_path=env_path)
    print("Loaded .env file")
    
    # Check if PINECONE_API_KEY is in environment
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if pinecone_key:
        print(f"PINECONE_API_KEY found: {pinecone_key[:5]}...")
    else:
        print("PINECONE_API_KEY not found in environment")
        
        # Debug: print all environment variables
        print("\nAll environment variables:")
        for key, value in os.environ.items():
            # Don't print the full values of sensitive keys
            if "KEY" in key or "TOKEN" in key:
                print(f"{key}: {value[:5]}...")
            else:
                print(f"{key}: {value}")
                
        # Try reading the .env file line by line
        print("\nContent of .env file:")
        with open(env_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "=" in line and not line.strip().startswith("#"):
                    key = line.split("=")[0].strip()
                    value = line.split("=", 1)[1].strip()
                    if "KEY" in key or "TOKEN" in key:
                        print(f"{key}: {value[:5]}...")
                    else:
                        print(f"{key}: {value}")
