import edge_tts
import re


# Text preprocessing
def preprocess_text(text):
    """Clean and format text."""
    text = re.sub(r"[#*_`\[\]]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def stream_tts(text):
    """Generate TTS audio - for backward compatibility."""
    processed_text = preprocess_text(text)

    if not processed_text:
        return

    # Use edge-tts with slower speed for clearer speech
    # Use edge-tts at normal speed for faster response
    communicate = edge_tts.Communicate(
        text=processed_text,
        voice="vi-VN-NamMinhNeural",
        rate="+0%",
        pitch="+0Hz",
        volume="+0%",
    )

    # Collect ALL audio data first
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])

    # Combine all chunks
    full_audio = b"".join(audio_chunks)

    # Stream in chunks
    chunk_size = 2048
    for i in range(0, len(full_audio), chunk_size):
        yield full_audio[i : i + chunk_size]


async def generate_full_tts(text):
    """Generate full TTS audio and return as single bytes."""
    processed_text = preprocess_text(text)

    if not processed_text:
        return b""

    # Use edge-tts with slower speed for clearer speech
    # Use edge-tts at normal speed for faster response
    communicate = edge_tts.Communicate(
        text=processed_text,
        voice="vi-VN-NamMinhNeural",
        rate="+0%",
        pitch="+0Hz",
        volume="+0%",
    )

    # Collect ALL audio data
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])

    # Return combined audio
    return b"".join(audio_chunks)
