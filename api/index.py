import json
import sys
import os
import base64
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from asr import transcribe_audio
except Exception as e:
    logger.warning(f"Failed to import asr: {e}")
    transcribe_audio = None

try:
    from brain import get_reply
except Exception as e:
    logger.warning(f"Failed to import brain: {e}")
    get_reply = None

try:
    from tts import generate_full_tts
except Exception as e:
    logger.warning(f"Failed to import tts: {e}")
    generate_full_tts = None

try:
    from lipsync import generate_visemes, estimate_word_timing
except Exception as e:
    logger.warning(f"Failed to import lipsync: {e}")
    generate_visemes = None
    estimate_word_timing = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
@app.post("/api")
async def handle_request(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {str(e)}"})

    text = body.get("text", "")
    audio_data = body.get("audio", "")
    exact_tts = body.get("exact_tts", False)

    try:
        if audio_data and transcribe_audio:
            try:
                audio_bytes = base64.b64decode(audio_data)
                text = transcribe_audio(audio_bytes)
                logger.info(f"Transcribed audio to text: {text[:50] if text else '(empty)'}...")
            except Exception as e:
                logger.error(f"ASR transcription error: {e}")
                return JSONResponse(status_code=400, content={"error": f"Audio transcription failed: {str(e)}"})

        if not text:
            return JSONResponse(status_code=400, content={"error": "Missing text or audio"})

        if exact_tts:
            reply = text
        elif get_reply:
            reply = get_reply(text)
        else:
            reply = text

        audio_bytes = b""
        viseme_sequence = []
        word_timing = []
        audio_duration_ms = 0

        if generate_full_tts:
            try:
                audio_bytes = await generate_full_tts(reply)
                if audio_bytes and generate_visemes and estimate_word_timing:
                    try:
                        viseme_sequence = generate_visemes(audio_bytes)
                        audio_duration_ms = len(audio_bytes) * 1000 // 24000
                        word_timing = estimate_word_timing(reply, audio_duration_ms)
                    except Exception as e:
                        logger.error(f"Lipsync error: {e}")
            except Exception as e:
                logger.error(f"TTS error: {e}")
                audio_bytes = b""

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else ""

        return JSONResponse(content={
            "text": reply,
            "user_text": text,
            "audio": audio_base64,
            "lip_sync": {
                "visemes": viseme_sequence,
                "words": word_timing,
                "duration": audio_duration_ms,
            },
        })

    except Exception as e:
        logger.error(f"API error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
