import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

_tts_state = {
    "ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Mark service ready at startup."""
    _tts_state["ready"] = True
    yield
    _tts_state["ready"] = False


app = FastAPI(
    title="TTS Service",
    description="OpenAI-compatible text-to-speech using ElevenLabs with edge-tts fallback when not configured.",
    version="1.1.0",
    lifespan=lifespan,
)

_DEFAULT_EN_VOICE = "en-US-JennyNeural"
_DEFAULT_AR_VOICE = "ar-EG-ShakirNeural"
_DEFAULT_ELEVENLABS_VOICE = "CwhRBWXzGAHq8TQ4Fs17"
_ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
_VOICE_RETRY_STATUSES = {401, 403, 404}
_PCM_RESPONSE_FORMAT = "pcm"
_MP3_RESPONSE_FORMAT = "mp3"
_ELEVENLABS_OUTPUT_FORMATS = {
    _MP3_RESPONSE_FORMAT: "mp3_44100_128",
    _PCM_RESPONSE_FORMAT: "pcm_24000",
}
_RESPONSE_MEDIA_TYPES = {
    _MP3_RESPONSE_FORMAT: "audio/mpeg",
    _PCM_RESPONSE_FORMAT: "application/octet-stream",
}
_RESPONSE_FILENAMES = {
    _MP3_RESPONSE_FORMAT: "speech.mp3",
    _PCM_RESPONSE_FORMAT: "speech.pcm",
}


class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize into speech.")
    model: Optional[str] = Field(None, description="OpenAI-compatible model hint. ElevenLabs uses eleven_multilingual_v2.")
    voice: Optional[str] = Field(None, description="ElevenLabs voice id when ElevenLabs is configured; edge-tts voice otherwise.")
    response_format: Optional[str] = Field(
        "mp3",
        description="Audio format. Supports mp3 and OpenAI-compatible raw 24 kHz mono PCM.",
    )
    speed: Optional[float] = Field(1.0, description="Speaking speed multiplier. Reserved for compatibility.")


def clean_for_tts(text: str) -> str:
    """Remove markdown formatting, bullet points, and extra whitespace."""
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Detect if text is primarily Arabic (>20% Arabic chars)."""
    compact = text.replace(" ", "")
    if not compact:
        return "en"
    arabic_chars = re.findall(r"[\u0600-\u06FF]", text)
    if len(arabic_chars) / len(compact) > 0.2:
        return "ar"
    return "en"


def split_text(text: str, max_len: int = 200) -> list[str]:
    """Split long text into ElevenLabs-sized sentence chunks."""
    if len(text) <= max_len:
        return [text]

    sentences = re.split(r"([.،!?؟])", text)
    chunks: list[str] = []
    current = ""
    for index in range(0, len(sentences), 2):
        sentence = sentences[index].strip()
        punctuation = sentences[index + 1] if index + 1 < len(sentences) else "."
        if not sentence:
            continue
        candidate = f"{sentence}{punctuation}"
        if len(current) + len(candidate) + 1 <= max_len:
            current = f"{current} {candidate}".strip()
        else:
            if current:
                chunks.append(current)
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def _elevenlabs_api_key() -> str | None:
    key = os.getenv("ELEVENLABS_API_KEY")
    return key.strip() if key and key.strip() else None


@dataclass
class ElevenLabsError(RuntimeError):
    status_code: int | None
    message: str
    retryable_with_voice: bool = False

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"HTTP {self.status_code}: {self.message}"


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _configured_elevenlabs_voice_candidates(requested_voice: Optional[str], language: str) -> list[str]:
    """Return configured voice ids/names in preference order without calling ElevenLabs."""
    candidates: list[str] = []
    if requested_voice and requested_voice.strip():
        candidates.append(requested_voice.strip())

    language_keys = (
        ["ELEVENLABS_AR_VOICE_ID", "ELEVENLABS_AR_VOICE", "ELEVENLABS_AR_VOICE_NAME"]
        if language == "ar"
        else ["ELEVENLABS_EN_VOICE_ID", "ELEVENLABS_EN_VOICE", "ELEVENLABS_EN_VOICE_NAME"]
    )
    common_keys = [
        "ELEVENLABS_VOICE_ID",
        "ELEVENLABS_VOICE",
        "ELEVENLABS_VOICE_NAME",
    ]
    for key in [*language_keys, *common_keys]:
        value = _env_value(key)
        if value:
            candidates.append(value)

    candidates.append(_DEFAULT_ELEVENLABS_VOICE)
    return _dedupe(candidates)


def _elevenlabs_headers(api_key: str, accept: str = "application/json") -> dict[str, str]:
    return {
        "xi-api-key": api_key,
        "Accept": accept,
    }


def _elevenlabs_voices(api_key: str) -> list[dict]:
    """Fetch accessible ElevenLabs voices. Response contents are non-secret."""
    import requests

    response = requests.get(
        _ELEVENLABS_VOICES_URL,
        headers=_elevenlabs_headers(api_key),
        timeout=30,
    )
    if response.status_code != 200:
        raise ElevenLabsError(response.status_code, "voices lookup failed")
    payload = response.json()
    voices = payload.get("voices", []) if isinstance(payload, dict) else []
    return [voice for voice in voices if isinstance(voice, dict)]


def _expand_with_accessible_voice_ids(candidates: list[str], api_key: str) -> list[str]:
    """Map configured names to ids and prefer accessible fallbacks from /v1/voices.

    ElevenLabs returns HTTP 401 for several conditions, including invalid voice
    access and exhausted quota. To make rebuilds stable when local env voice ids
    drift, validate configured voice ids/names against the account's accessible
    voices before synthesizing. If no configured candidate is accessible, try an
    account voice first and keep the configured values only as late fallbacks.
    """
    try:
        voices = _elevenlabs_voices(api_key)
    except ElevenLabsError:
        return candidates

    lowered = {candidate.lower(): candidate for candidate in candidates}
    matched_ids: list[str] = []
    fallback_ids: list[str] = []
    for voice in voices:
        voice_id = str(voice.get("voice_id") or "").strip()
        voice_name = str(voice.get("name") or "").strip()
        if not voice_id:
            continue
        if voice_id in candidates or (voice_name and voice_name.lower() in lowered):
            matched_ids.append(voice_id)
        else:
            fallback_ids.append(voice_id)

    if matched_ids:
        return _dedupe([*matched_ids, *fallback_ids, *candidates])
    return _dedupe([*fallback_ids, *candidates])


def _elevenlabs_error_from_response(response) -> ElevenLabsError:
    provider_status = ""
    provider_message = ""
    try:
        detail = response.json().get("detail", {})
        if isinstance(detail, dict):
            provider_status = str(detail.get("status") or "").strip()
            provider_message = str(detail.get("message") or "").strip()
    except Exception:
        provider_status = ""
        provider_message = ""

    if provider_status == "quota_exceeded":
        return ElevenLabsError(
            response.status_code,
            "quota_exceeded: ElevenLabs account has no remaining credits for text-to-speech",
            retryable_with_voice=False,
        )

    message = "text-to-speech request failed"
    if provider_status:
        message = f"{provider_status}: {message}"
    elif provider_message:
        message = provider_message[:160]
    return ElevenLabsError(
        response.status_code,
        message,
        retryable_with_voice=response.status_code in _VOICE_RETRY_STATUSES,
    )


def _resolve_edge_voice(text: str, requested_voice: Optional[str]) -> str:
    if requested_voice:
        return requested_voice
    return _DEFAULT_AR_VOICE if detect_language(text) == "ar" else _DEFAULT_EN_VOICE


def _normalize_response_format(response_format: Optional[str]) -> str:
    normalized = (response_format or _MP3_RESPONSE_FORMAT).strip().lower()
    if normalized in {"mpeg", "mp3"}:
        return _MP3_RESPONSE_FORMAT
    if normalized in {"pcm", "s16le"}:
        return _PCM_RESPONSE_FORMAT
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="response_format must be one of: mp3, pcm.",
    )


def _elevenlabs_audio_for_voice(
    text: str,
    voice_id: str,
    api_key: str,
    response_format: str,
) -> bytes:
    """Synthesize text with a specific ElevenLabs voice and return audio bytes.

    LiveKit's OpenAI TTS adapter requests response_format=pcm and then treats
    the response body as raw 24 kHz mono signed 16-bit PCM. Returning MP3 for
    that request sounds like static, so forward the matching ElevenLabs
    output_format instead of relying on the provider default.
    """
    import requests

    headers = _elevenlabs_headers(
        api_key,
        accept="application/octet-stream" if response_format == _PCM_RESPONSE_FORMAT else "audio/mpeg",
    )
    headers["Content-Type"] = "application/json"
    output_format = _ELEVENLABS_OUTPUT_FORMATS[response_format]
    audio_parts: list[bytes] = []
    for chunk in split_text(text):
        payload = {
            "text": chunk,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.65,
                "similarity_boost": 0.85,
                "style": 0.25,
                "use_speaker_boost": True,
            },
        }
        response = requests.post(
            _ELEVENLABS_TTS_URL.format(voice_id=quote(voice_id, safe="")),
            params={"output_format": output_format},
            json=payload,
            headers=headers,
            timeout=60,
        )
        if response.status_code != 200:
            raise _elevenlabs_error_from_response(response)
        if not response.content:
            raise ElevenLabsError(response.status_code, "empty audio response")
        audio_parts.append(response.content)
    return b"".join(audio_parts)


def _elevenlabs_audio(
    text: str,
    requested_voice: Optional[str],
    language: str,
    response_format: str,
) -> bytes:
    """Synthesize text with configured ElevenLabs voice selection and safe fallback."""
    api_key = _elevenlabs_api_key()
    if not api_key:
        raise ElevenLabsError(None, "ELEVENLABS_API_KEY is not configured")

    voice_candidates = _configured_elevenlabs_voice_candidates(requested_voice, language)
    last_error: ElevenLabsError | None = None
    looked_up_voices = False

    if not requested_voice:
        voice_candidates = _expand_with_accessible_voice_ids(voice_candidates, api_key)
        looked_up_voices = True
    failed_voices: set[str] = set()

    index = 0
    while index < len(voice_candidates):
        voice_id = voice_candidates[index]
        if voice_id in failed_voices:
            index += 1
            continue
        try:
            return _elevenlabs_audio_for_voice(text, voice_id, api_key, response_format)
        except ElevenLabsError as exc:
            failed_voices.add(voice_id)
            last_error = exc
            if not exc.retryable_with_voice:
                raise
            if not looked_up_voices:
                looked_up_voices = True
                voice_candidates = _expand_with_accessible_voice_ids(voice_candidates, api_key)
                index = -1
        index += 1

    if last_error:
        raise last_error
    raise ElevenLabsError(None, "no ElevenLabs voice candidates available")


async def _edge_audio_stream(text: str, voice: str) -> AsyncGenerator[bytes, None]:
    """Yield MP3 audio chunks from edge-tts."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            yield chunk["data"]


async def _single_audio_chunk(audio: bytes) -> AsyncGenerator[bytes, None]:
    yield audio


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Kubernetes-style health probe."""
    healthy = _tts_state["ready"] is True
    if not healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "service": "tts"},
        )
    return {"status": "ok", "service": "tts", "provider": "elevenlabs" if _elevenlabs_api_key() else "edge-tts"}


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """OpenAI-compatible audio speech endpoint."""
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

    language = detect_language(clean_text)
    response_format = _normalize_response_format(request.response_format)
    if _elevenlabs_api_key():
        try:
            audio = _elevenlabs_audio(clean_text, request.voice, language, response_format)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"ElevenLabs TTS failed: {exc}",
            ) from exc
        body = _single_audio_chunk(audio)
    else:
        if response_format == _PCM_RESPONSE_FORMAT:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="response_format=pcm requires ElevenLabs TTS configuration.",
            )
        voice = _resolve_edge_voice(clean_text, request.voice)
        body = _edge_audio_stream(clean_text, voice)

    return StreamingResponse(
        body,
        media_type=_RESPONSE_MEDIA_TYPES[response_format],
        headers={
            "Content-Disposition": f'attachment; filename="{_RESPONSE_FILENAMES[response_format]}"',
        },
    )
