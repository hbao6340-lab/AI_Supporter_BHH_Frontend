import json
import sys
import os
import base64
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr import transcribe_audio
from brain import get_reply
from tts import preprocess_text, generate_full_tts, stream_tts
from lipsync import generate_visemes, estimate_word_timing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/stream")
async def handle_stream(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {str(e)}"})

    text = body.get("text", "")
    exact_tts = body.get("exact_tts", False)
    audio_data = body.get("audio", "")

    try:
        if audio_data:
            try:
                audio_bytes = base64.b64decode(audio_data)
                text = transcribe_audio(audio_bytes)
                logger.info(f"Transcribed audio to text: {text[:50] if text else '(empty)'}...")
            except Exception as e:
                logger.error(f"ASR transcription error: {e}")
                return JSONResponse(status_code=400, content={"error": f"Audio transcription failed: {str(e)}"})

        if not text:
            return JSONResponse(status_code=400, content={"error": "Missing text"})

        if exact_tts:
            reply = text
        else:
            reply = get_reply(text)

        async def event_stream():
            if audio_data:
                yield f"data: {json.dumps({'user_text': text})}\n\n"

            yield f"data: {json.dumps({'text': reply})}\n\n"

            communicate = None
            try:
                import edge_tts
                processed_text = preprocess_text(reply)
                if processed_text:
                    communicate = edge_tts.Communicate(
                        text=processed_text,
                        voice="vi-VN-NamMinhNeural",
                        rate="+0%",
                        pitch="+0Hz",
                        volume="+0%",
                    )

                    audio_chunks = []
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_b64 = base64.b64encode(chunk["data"]).decode("utf-8")
                            yield f"data: {json.dumps({'audio': audio_b64})}\n\n"
                            audio_chunks.append(chunk["data"])
                        elif chunk["type"] == "word":
                            yield f"data: {json.dumps({'word': chunk['data']})}\n\n"

                    if audio_chunks and False:
                        full_audio = b"".join(audio_chunks)
                        visemes = generate_visemes(full_audio)
                        audio_duration_ms = len(full_audio) * 1000 // 24000
                        word_timing = estimate_word_timing(reply, audio_duration_ms)
                        yield f"data: {json.dumps({'visemes': visemes, 'word_timing': word_timing, 'duration': audio_duration_ms})}\n\n"
            except ImportError:
                logger.warning("edge_tts not available, skipping streaming TTS")

            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Stream error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})