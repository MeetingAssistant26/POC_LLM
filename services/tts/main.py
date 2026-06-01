import os
import re
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Singleton state for edge-tts readiness
_tts_state = {
    "ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Mark service ready at startup (edge-tts has no heavy init)."""
    _tts_state["ready"] = True
    yield
    _tts_state["ready"] = False


app = FastAPI(
    title="TTS Service",
    description="OpenAI-compatible text-to-speech using edge-tts (Microsoft Azure TTS edge endpoint).",
    version="1.0.0",
    lifespan=lifespan,
)

# Voice defaults
_DEFAULT_EN_VOICE = "en-US-JennyNeural"
_DEFAULT_AR_VOICE = "ar-EG-ShakirNeural"


class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize into speech.")
    model: Optional[str] = Field(None, description="Ignored; always uses edge-tts.")
    voice: Optional[str] = Field(None, description="Edge-tts voice name. Auto-detected if omitted.")
    response_format: Optional[str] = Field("mp3", description="Audio format. Only mp3 is supported.")
    speed: Optional[float] = Field(1.0, description="Speaking speed multiplier (not implemented by edge-tts).")


def clean_for_tts(text: str) -> str:
    """Remove markdown formatting, bullet points, and extra whitespace."""
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detect_language(text: str) -> str:
    """Detect if text is primarily Arabic (>30% Arabic chars)."""
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    if len(arabic_chars) > len(text) * 0.3:
        return "ar"
    return "en"


def _resolve_voice(text: str, requested_voice: Optional[str]) -> str:
    """Return the voice to use: requested if provided, otherwise auto-detected."""
    if requested_voice:
        return requested_voice
    lang = detect_language(text)
    return _DEFAULT_AR_VOICE if lang == "ar" else _DEFAULT_EN_VOICE


async def _audio_stream(text: str, voice: str) -> AsyncGenerator[bytes, None]:
    """Yield MP3 audio chunks from edge-tts."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            yield chunk["data"]


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Kubernetes-style health probe."""
    healthy = _tts_state["ready"] is True
    if not healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "service": "tts"},
        )
    return {"status": "ok", "service": "tts"}


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    OpenAI-compatible audio speech endpoint.

    - **input**: Text to synthesize.
    - **model**: Ignored; always uses edge-tts.
    - **voice**: Edge-tts voice name (e.g. en-US-JennyNeural). Auto-detected if omitted.
    - **response_format**: Only mp3 is supported.
    - **speed**: Ignored; edge-tts does not expose speed control.
    """
    if not request.input or not request.input.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="input field is required and must not be empty.",
        )

    clean_text = clean_for_tts(request.input)
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="input contains no usable text after cleaning.",
        )

    voice = _resolve_voice(clean_text, request.voice)

    return StreamingResponse(
        _audio_stream(clean_text, voice),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": 'attachment; filename="speech.mp3"',
        },
    )
