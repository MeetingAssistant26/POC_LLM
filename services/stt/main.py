import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
import torch
import whisperx
from fastapi import FastAPI, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup; keep it hot for the lifetime of the process."""
    device = "cpu"
    compute_type = "int8"
    _load_model_singleton(device, compute_type)
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
        from whisperx.diarize import DiarizationPipeline

        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        diarize_segments = diarize_model(audio_path)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    return result


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Kubernetes-style health probe."""
    healthy = _stt_state["model"] is not None
    if not healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "model_loaded": False},
        )
    return {"status": "ok", "model_loaded": True}


@app.post("/v1/audio/transcriptions")
async def create_transcription(
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
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = _run_pipeline(tmp_path, language=language)
        return _format_openai_response(result, response_format)
    finally:
        os.unlink(tmp_path)
