
import os
import sys
from google import genai
from google.genai import types

def test_connection():
    project = os.environ.get("VERTEX_PROJECT")
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    
    print(f"Testing Vertex AI connection...")
    print(f"Project: [{project}]")
    print(f"Location: [{location}]")
    
    try:
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location
        )
        
        print("Constructed client. Attempting count_tokens...")
        response = client.models.count_tokens(
            model="gemini-2.0-flash",
            contents="Hello"
        )
        print(f"Success! Total tokens: {response.total_tokens}")
    except Exception as e:
        print(f"\nCaught Exception: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        if hasattr(e, 'status_code'):
            print(f"Status code: {e.status_code}")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Simulate the env loading
    from dotenv import load_dotenv
    load_dotenv("/home/default/.hermes/.env")
    
    test_connection()
