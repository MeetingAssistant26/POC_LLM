import asyncio
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
import torch
import whisperx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from services.ai_debug import duration_ms, parse_trace_context, post_trace_event

# Singleton state for the loaded model and device
_stt_state = {
    "model": None,
    "device": None,
    "compute_type": None,
    "hf_token": None,
}


def _load_model_singleton(device: str, compute_type: str):
    """Load WhisperX model once and cache it globally."""
    if _stt_state["model"] is None:
        print(f"[STT] Loading WhisperX model 'medium' on {device} ({compute_type}) ...")
        _stt_state["model"] = whisperx.load_model(
            "medium",
            device,
            compute_type=compute_type,
        )
        _stt_state["device"] = device
        _stt_state["compute_type"] = compute_type
        _stt_state["hf_token"] = os.environ.get("HF_TOKEN")
        print("[STT] Model loaded successfully.")
    return _stt_state["model"]


async def _load_model_async(device: str, compute_type: str):
    """Load the WhisperX model in a background thread."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model_singleton, device, compute_type)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start model loading in the background; yield immediately so uvicorn accepts requests."""
    device = "cpu"
    compute_type = "int8"
    asyncio.create_task(_load_model_async(device, compute_type))
    yield
    # Optional cleanup on shutdown
    _stt_state["model"] = None


app = FastAPI(
    title="STT Service",
    description="OpenAI-compatible speech-to-text using WhisperX with optional speaker diarization.",
    version="1.0.0",
    lifespan=lifespan,
)


def _format_openai_response(result: dict, response_format: str = "json") -> dict:
    """Convert WhisperX result into OpenAI-compatible shapes."""
    segments = result.get("segments", [])
    full_text = " ".join(seg.get("text", "").strip() for seg in segments).strip()

    if response_format == "text":
        return {"text": full_text}

    if response_format == "srt":
        srt_lines = []
        for i, seg in enumerate(segments, start=1):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "").strip()
            srt_lines.append(f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
        return {"text": "\n".join(srt_lines)}

    if response_format == "vtt":
        vtt_lines = ["WEBVTT\n"]
        for seg in segments:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "").strip()
            vtt_lines.append(f"{_vtt_time(start)} --> {_vtt_time(end)}\n{text}\n")
        return {"text": "\n".join(vtt_lines)}

    # verbose_json and json default
    out_segments = []
    for seg in segments:
        out_seg = {
            "id": seg.get("id", len(out_segments)),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip(),
        }
        speaker = seg.get("speaker")
        if speaker:
            out_seg["speaker"] = speaker
        out_segments.append(out_seg)

    resp: dict = {
        "text": full_text,
        "segments": out_segments,
        "language": result.get("language", "en"),
    }
    return resp


def _srt_time(seconds: float) -> str:
    """Format seconds as SRT HH:MM:SS,mmm"""
    millis = int((seconds % 1) * 1000)
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _vtt_time(seconds: float) -> str:
    """Format seconds as WebVTT HH:MM:SS.mmm"""
    millis = int((seconds % 1) * 1000)
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def _build_diarization_pipeline(hf_token: str, device: str):
    """Create a WhisperX diarization pipeline across supported constructor APIs.

    WhisperX has changed this constructor over time. Some versions accept
    `use_auth_token=...`, while older/newer builds may not accept the previous
    `token=...` keyword. Diarization is optional for this service, so callers can
    skip it when the installed API is incompatible and still return the
    transcription.
    """
    from whisperx.diarize import DiarizationPipeline

    constructor_attempts = (
        {"use_auth_token": hf_token, "device": device},
        {"token": hf_token, "device": device},
        {"device": device},
    )
    errors = []

    for kwargs in constructor_attempts:
        try:
            return DiarizationPipeline(**kwargs)
        except TypeError as exc:
            errors.append(f"{kwargs}: {exc}")

    raise TypeError("No compatible DiarizationPipeline constructor found. " + " | ".join(errors))


def _run_pipeline(audio_path: str, language: Optional[str] = None) -> dict:
    """Run the full WhisperX pipeline: transcribe → align → (diarize)."""
    model = _stt_state["model"]
    device = _stt_state["device"]
    hf_token = _stt_state["hf_token"]

    audio = whisperx.load_audio(audio_path)

    # 1. Transcription
    result = model.transcribe(audio, language=language)

    # 2. Alignment
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
    )

    # 3. Speaker diarization (optional)
    if hf_token:
        try:
            diarize_model = _build_diarization_pipeline(hf_token, device)
            diarize_segments = diarize_model(audio_path)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as exc:
            print(f"[STT] Speaker diarization skipped; transcription preserved. Reason: {exc}")

    return result


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Kubernetes-style liveness probe — returns 200 as long as the server is running."""
    return {"status": "ok", "model_loaded": _stt_state["model"] is not None}


@app.get("/readyz")
async def readyz():
    """Kubernetes-style readiness probe — returns 200 only when the model is loaded."""
    if _stt_state["model"] is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "model_loaded": False},
        )
    return {"status": "ok", "model_loaded": True}


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
):
    """
    OpenAI-compatible audio transcription endpoint.

    - **file**: Audio file to transcribe (wav, mp3, etc.)
    - **model**: Ignored; always uses WhisperX medium.
    - **language**: Optional ISO language code.
    - **response_format**: `json`, `verbose_json`, `text`, `srt`, `vtt`.
    """
    trace_ctx = parse_trace_context(request.headers)
    start = time.perf_counter()

    if _stt_state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="STT model is not ready yet. Please retry shortly.",
        )
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = _run_pipeline(tmp_path, language=language)
        response_body = _format_openai_response(result, response_format)
        step = {
            "type": "stt",
            "provider": "WhisperX",
            "endpoint": str(request.url),
            "model": "medium",
            "durationMs": duration_ms(start),
        }
        if trace_ctx and trace_ctx.persist_payloads:
            step["text"] = response_body.get("text")
            step["responsePayload"] = response_body
        await post_trace_event(trace_ctx, "stt_completed", step=step)
        return response_body
    finally:
        os.unlink(tmp_path)
