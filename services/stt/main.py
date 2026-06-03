import asyncio
import base64
import json
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
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

STT_PROVIDER_LOCAL = "local"
STT_PROVIDER_OPENAI_COMPATIBLE = "openai-compatible"


def _stt_provider() -> str:
    return os.environ.get("STT_PROVIDER", STT_PROVIDER_LOCAL).strip().lower() or STT_PROVIDER_LOCAL


def _upstream_base_url() -> str:
    return os.environ.get("STT_UPSTREAM_BASE_URL", "").strip().rstrip("/")


def _upstream_api_key() -> str:
    return os.environ.get("STT_UPSTREAM_API_KEY", "").strip()


def _upstream_model() -> str:
    return os.environ.get("STT_UPSTREAM_MODEL", "").strip()


def _upstream_request_format() -> str:
    configured = os.environ.get("STT_UPSTREAM_REQUEST_FORMAT", "auto").strip().lower() or "auto"
    if configured == "auto" and "openrouter.ai" in _upstream_base_url():
        model = _upstream_model().lower()
        if "voxtral" in model and "transcribe" not in model:
            return "openrouter-chat-audio"
        return "openrouter-json"
    return configured


def _upstream_language() -> str:
    return os.environ.get("STT_UPSTREAM_LANGUAGE", "").strip()


def _resolved_upstream_language(data: dict) -> str | None:
    """Resolve the language sent to an upstream STT provider.

    The LiveKit OpenAI STT plugin defaults to sending language=en. For providers
    with useful language auto-detection, callers can set STT_UPSTREAM_LANGUAGE=auto
    to intentionally omit the upstream language field instead of forwarding that
    default English hint.
    """
    configured = _upstream_language()
    if configured:
        if configured.lower() in {"auto", "detect", "auto-detect", "auto_detect"}:
            return None
        return configured

    incoming = str(data.get("language") or "").strip()
    return incoming or None


def _provider_config_status() -> dict:
    provider = _stt_provider()
    if provider == STT_PROVIDER_LOCAL:
        return {
            "provider": provider,
            "configured": True,
            "missing": [],
        }

    if provider == STT_PROVIDER_OPENAI_COMPATIBLE:
        missing = []
        if not _upstream_base_url():
            missing.append("STT_UPSTREAM_BASE_URL")
        if not _upstream_api_key():
            missing.append("STT_UPSTREAM_API_KEY")
        request_format = _upstream_request_format()
        error = None
        if request_format not in {"auto", "openai-multipart", "openrouter-json", "openrouter-chat-audio"}:
            error = (
                f"Unsupported STT_UPSTREAM_REQUEST_FORMAT '{request_format}'. "
                "Use 'auto', 'openai-multipart', 'openrouter-json', or 'openrouter-chat-audio'."
            )
        return {
            "provider": provider,
            "configured": len(missing) == 0 and error is None,
            "missing": missing,
            "request_format": request_format,
            "error": error,
        }

    return {
        "provider": provider,
        "configured": False,
        "missing": ["STT_PROVIDER"],
        "error": f"Unsupported STT_PROVIDER '{provider}'. Use 'local' or 'openai-compatible'.",
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
    if _stt_provider() == STT_PROVIDER_LOCAL:
        device = "cpu"
        compute_type = "int8"
        asyncio.create_task(_load_model_async(device, compute_type))
    yield
    # Optional cleanup on shutdown
    _stt_state["model"] = None


app = FastAPI(
    title="STT Service",
    description="OpenAI-compatible speech-to-text using local WhisperX or an OpenAI-compatible upstream proxy.",
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
    """Kubernetes-style liveness probe with provider configuration status."""
    config = _provider_config_status()
    body = {
        "status": "ok" if config["configured"] else "not_configured",
        "provider": config["provider"],
        "model_loaded": _stt_state["model"] is not None,
        "upstream_configured": config["configured"] if config["provider"] == STT_PROVIDER_OPENAI_COMPATIBLE else None,
    }
    if config.get("missing"):
        body["missing"] = config["missing"]
    if config.get("error"):
        body["error"] = config["error"]
    if not config["configured"]:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body)
    return body


@app.get("/readyz")
async def readyz():
    """Kubernetes-style readiness probe — returns 200 only when the model is loaded."""
    config = _provider_config_status()
    if not config["configured"]:
        body = {
            "status": "not_ready",
            "provider": config["provider"],
            "model_loaded": _stt_state["model"] is not None,
            "missing": config.get("missing", []),
        }
        if config.get("error"):
            body["error"] = config["error"]
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=body,
        )

    if config["provider"] == STT_PROVIDER_OPENAI_COMPATIBLE:
        return {"status": "ok", "provider": config["provider"], "upstream_configured": True}

    if _stt_state["model"] is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "provider": config["provider"], "model_loaded": False},
        )
    return {"status": "ok", "provider": config["provider"], "model_loaded": True}


def _post_upstream_transcription(upstream_url, headers, data, files):
    """Post multipart audio to the upstream STT provider.

    httpx 0.28 builds multipart uploads as a sync request stream. Sending that
    stream through AsyncClient raises "Attempted to send a sync request with an
    AsyncClient instance", so the async FastAPI handler runs this sync client in
    a worker thread.
    """
    with httpx.Client(timeout=180.0) as client:
        return client.post(
            upstream_url,
            headers=headers,
            data=dict(data),
            files=files,
        )


def _post_upstream_transcription_json(upstream_url, headers, payload):
    with httpx.Client(timeout=180.0) as client:
        return client.post(
            upstream_url,
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
        )


def _audio_format(filename: str | None, content_type: str | None) -> str:
    name = (filename or "").lower()
    for extension, audio_format in (
        (".wav", "wav"),
        (".mp3", "mp3"),
        (".flac", "flac"),
        (".m4a", "m4a"),
        (".aac", "aac"),
        (".ogg", "ogg"),
    ):
        if name.endswith(extension):
            return audio_format
    if content_type:
        subtype = content_type.split("/", 1)[-1].split(";", 1)[0].lower()
        if subtype in {"wav", "mpeg", "mp3", "flac", "x-m4a", "aac", "ogg"}:
            return "mp3" if subtype == "mpeg" else subtype.removeprefix("x-")
    return "wav"


def _chat_audio_content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict) and part.get("text")
        ).strip()
    return ""


def _is_likely_empty_audio_hallucination(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not normalized:
        return False
    boilerplate_markers = (
        "hello everyone welcome to the podcast",
        "today we have a special guest",
        "let s get started",
        "thanks for watching",
        "don t forget to subscribe",
    )
    marker_hits = sum(1 for marker in boilerplate_markers if marker in normalized)
    return marker_hits >= 2 or (
        "welcome to the podcast" in normalized and "special guest" in normalized
    )


def _suppress_empty_audio_hallucination(response_body):
    if not isinstance(response_body, dict):
        return response_body
    text = str(response_body.get("text") or "")
    if not _is_likely_empty_audio_hallucination(text):
        return response_body
    sanitized = dict(response_body)
    sanitized["text"] = ""
    sanitized["segments"] = []
    sanitized["filtered_reason"] = "likely_empty_audio_hallucination"
    return sanitized


def _parse_chat_audio_transcription(response_body):
    choices = response_body.get("choices") if isinstance(response_body, dict) else None
    message = choices[0].get("message", {}) if choices else {}
    raw = _chat_audio_content_text(message.get("content"))
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    segments = []
    try:
        parsed = json.loads(cleaned)
        for item in parsed.get("segments", []) if isinstance(parsed, dict) else []:
            text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
            if not text:
                continue
            start = item.get("t", item.get("start", 0)) if isinstance(item, dict) else 0
            try:
                start = float(start)
            except (TypeError, ValueError):
                start = 0.0
            segments.append({"id": len(segments), "start": start, "end": start, "text": text})
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    if not segments and raw:
        text = re.sub(r"\s+", " ", raw).strip()
        if text:
            segments.append({"id": 0, "start": 0.0, "end": 0.0, "text": text})

    combined_text = " ".join(segment["text"] for segment in segments).strip()
    if _is_likely_empty_audio_hallucination(combined_text):
        segments = []
        combined_text = ""

    return {
        "text": combined_text,
        "segments": segments,
        "usage": response_body.get("usage") if isinstance(response_body, dict) else None,
    }


async def _proxy_transcription_to_openai_compatible(
    request: Request,
    file: UploadFile,
    request_model: str,
    trace_ctx,
    start: float,
):
    config = _provider_config_status()
    if not config["configured"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "STT upstream is not configured.",
                "missing": config.get("missing", []),
                "error": config.get("error"),
            },
        )

    form = await request.form()
    data = []
    has_response_format = False
    for key, value in form.multi_items():
        if key == "file" or isinstance(value, UploadFile):
            continue
        if key == "model":
            continue
        if key == "response_format":
            has_response_format = True
        data.append((key, str(value)))

    data.append(("model", _upstream_model() or request_model))
    if not has_response_format:
        data.append(("response_format", "verbose_json"))

    contents = await file.read()
    upstream_url = f"{_upstream_base_url()}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {_upstream_api_key()}"}
    data_dict = dict(data)

    try:
        request_format = _upstream_request_format()
        if request_format == "openrouter-json":
            payload = {
                "model": data_dict["model"],
                "input_audio": {
                    "data": base64.b64encode(contents).decode("ascii"),
                    "format": _audio_format(file.filename, file.content_type),
                },
            }
            language = _resolved_upstream_language(data_dict)
            if language:
                payload["language"] = language
            upstream_response = await asyncio.to_thread(
                _post_upstream_transcription_json,
                upstream_url,
                headers,
                payload,
            )
        elif request_format == "openrouter-chat-audio":
            upstream_url = f"{_upstream_base_url()}/chat/completions"
            payload = {
                "model": data_dict["model"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe the audio. Return ONLY valid JSON (no markdown). "
                                    'Schema: {"segments":[{"t":number,"text":string}]}. '
                                    "If the audio has no clear human speech, return {\"segments\":[]}. "
                                    "t = seconds from the start of this chunk (best-effort; use 0 if unsure). "
                                    "Keep segments short (phrases/sentences)."
                                ),
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64.b64encode(contents).decode("ascii"),
                                    "format": _audio_format(file.filename, file.content_type),
                                },
                            },
                        ],
                    }
                ],
                "stream": False,
            }
            upstream_response = await asyncio.to_thread(
                _post_upstream_transcription_json,
                upstream_url,
                headers,
                payload,
            )
        else:
            files = {
                "file": (
                    file.filename or "audio.wav",
                    contents,
                    file.content_type or "application/octet-stream",
                )
            }
            upstream_response = await asyncio.to_thread(
                _post_upstream_transcription,
                upstream_url,
                headers,
                data,
                files,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"STT upstream request failed: {exc.__class__.__name__}",
        ) from exc

    content_type = upstream_response.headers.get("content-type", "")
    if upstream_response.status_code >= 400:
        detail = upstream_response.text
        if "application/json" in content_type:
            try:
                detail = upstream_response.json()
            except ValueError:
                pass
        raise HTTPException(status_code=upstream_response.status_code, detail=detail)

    if "application/json" in content_type:
        response_body = upstream_response.json()
        if _upstream_request_format() == "openrouter-chat-audio":
            response_body = _parse_chat_audio_transcription(response_body)
        else:
            response_body = _suppress_empty_audio_hallucination(response_body)
        step = {
            "type": "stt",
            "provider": "OpenAICompatible",
            "endpoint": upstream_url,
            "model": _upstream_model() or request_model,
            "durationMs": duration_ms(start),
        }
        if trace_ctx and trace_ctx.persist_payloads:
            step["text"] = response_body.get("text") if isinstance(response_body, dict) else None
            step["responsePayload"] = response_body
        await post_trace_event(trace_ctx, "stt_completed", step=step)
        return JSONResponse(status_code=upstream_response.status_code, content=response_body)

    return JSONResponse(
        status_code=upstream_response.status_code,
        content={"text": upstream_response.text},
    )


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("verbose_json"),
):
    """
    OpenAI-compatible audio transcription endpoint.

    - **file**: Audio file to transcribe (wav, mp3, etc.)
    - **model**: Local mode ignores this and uses WhisperX medium; proxy mode forwards the configured upstream model or this value.
    - **language**: Optional ISO language code.
    - **response_format**: `json`, `verbose_json`, `text`, `srt`, `vtt`.
    """
    trace_ctx = parse_trace_context(request.headers)
    start = time.perf_counter()

    if _stt_provider() == STT_PROVIDER_OPENAI_COMPATIBLE:
        return await _proxy_transcription_to_openai_compatible(
            request,
            file,
            model,
            trace_ctx,
            start,
        )

    if _stt_provider() != STT_PROVIDER_LOCAL:
        config = _provider_config_status()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=config.get("error", "Unsupported STT provider."),
        )

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
