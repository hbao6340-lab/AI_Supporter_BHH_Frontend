"""
Speech-to-Text using OpenAI Whisper API
Uses OpenAI's API instead of local model to reduce package size
"""
import os
import io

# Try to use OpenAI Whisper API, fall back to placeholder if not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: OpenAI package not installed")

# Get API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("WARNING: OPENAI_API_KEY environment variable not set - speech transcription will not work")
else:
    print("INFO: OPENAI_API_KEY is configured")

def transcribe_audio(audio_bytes):
    """
    Transcribe audio bytes to text using OpenAI Whisper API.
    Falls back to returning empty string if API is not available.
    """
    if not audio_bytes or len(audio_bytes) == 0:
        print("WARNING: Empty audio bytes received for transcription")
        return ""
    
    # Check if OpenAI is available
    if not OPENAI_AVAILABLE:
        print("ERROR: OpenAI package not available - cannot transcribe audio")
        return ""
    
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not set - cannot transcribe audio")
        return ""
    
    try:
        client = OpenAI(api_key=openai_api_key)
        
        # Create a file-like object from bytes
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        
        # Use OpenAI Whisper API with context prompt for better Vietnamese transcription
        print("INFO: Calling OpenAI Whisper API for transcription...")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="vi",  # Vietnamese
            prompt="Đây là bản ghi âm tiếng Việt về các vấn đề hành chính của phường Tân Hưng. Vui lòng transcribe chính xác.",
            temperature=0.0  # More deterministic output
        )
        
        result = transcript.text
        print(f"INFO: Transcription successful: {result[:50]}...")
        return result
        
    except Exception as e:
        print(f"ERROR: Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return ""
