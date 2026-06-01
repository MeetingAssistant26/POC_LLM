#!/usr/bin/env python3
"""Smoke test for the AI microservices pipeline (STT -> LLM -> TTS).

Usage:
    docker compose up -d
    python3 scripts/smoke_test.py

The script sends a real speech WAV fixture to the STT service, forwards the
non-empty transcript to the LLM service, and finally sends a deliberately tiny
LLM-derived reply to the TTS service to avoid exhausting paid TTS quota. It
exits 0 when every step returns the expected HTTP status and payload shape.
"""

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

STT_URL = "http://localhost:8001/v1/audio/transcriptions"
LLM_URL = "http://localhost:8002/v1/chat/completions"
TTS_URL = "http://localhost:8003/v1/audio/speech"

HEALTH_URLS = {
    "stt": "http://localhost:8001/readyz",
    "llm": "http://localhost:8002/healthz",
    "tts": "http://localhost:8003/healthz",
}

POLL_INTERVAL = 5
POLL_MAX_WAIT = 180
DEFAULT_AUDIO_FIXTURE = Path("audio_clean/meeting_1.wav")
SMOKE_AUDIO_SECONDS = float(os.getenv("SMOKE_AUDIO_SECONDS", "20"))
SMOKE_TTS_MAX_CHARS = int(os.getenv("SMOKE_TTS_MAX_CHARS", "8"))


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:
        return ""


def _http_get_status(url: str, timeout: int = 5) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, str], bytes]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, dict(response.headers), response.read()


def _post_multipart_file(
    url: str,
    field_name: str,
    file_path: Path,
    filename: str,
    content_type: str,
    fields: dict[str, str],
    timeout: int,
) -> tuple[int, dict[str, str], bytes]:
    boundary = f"----ai-work-smoke-{int(time.time() * 1000)}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    body = b"".join(chunks)
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, dict(response.headers), response.read()


def _wait_for_services() -> None:
    """Poll health/readiness endpoints until all three services are up or timeout."""
    print("Waiting for services to be healthy ...")
    deadline = time.time() + POLL_MAX_WAIT
    while time.time() < deadline:
        all_ok = True
        for name, url in HEALTH_URLS.items():
            try:
                status = _http_get_status(url, timeout=5)
                if status == 200:
                    print(f"  {name}: healthy")
                else:
                    print(f"  {name}: not ready ({status})")
                    all_ok = False
            except urllib.error.HTTPError as exc:
                print(f"  {name}: not ready ({exc.code})")
                all_ok = False
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"  {name}: unreachable ({type(exc).__name__})")
                all_ok = False
        if all_ok:
            print("All services are healthy.\n")
            return
        time.sleep(POLL_INTERVAL)
    raise RuntimeError(f"Services did not become healthy within {POLL_MAX_WAIT}s")


def prepare_audio_fixture(path: Path) -> Path:
    """Copy a short clip from the configured real speech fixture for upload."""
    fixture = Path(os.getenv("SMOKE_STT_AUDIO", str(DEFAULT_AUDIO_FIXTURE)))
    if not fixture.exists():
        raise RuntimeError(f"STT audio fixture does not exist: {fixture}")
    if SMOKE_AUDIO_SECONDS <= 0:
        shutil.copyfile(fixture, path)
        return fixture
    with wave.open(str(fixture), "rb") as source:
        frames_to_copy = min(
            source.getnframes(),
            int(source.getframerate() * SMOKE_AUDIO_SECONDS),
        )
        params = source.getparams()
        frames = source.readframes(frames_to_copy)
    with wave.open(str(path), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)
    return fixture


def run_stt(wav_path: Path) -> str:
    print("Step 1: POST test WAV to STT (http://localhost:8001/v1/audio/transcriptions) ...")
    status, _headers, response_body = _post_multipart_file(
        STT_URL,
        field_name="file",
        file_path=wav_path,
        filename="test.wav",
        content_type="audio/wav",
        fields={"response_format": "json"},
        timeout=120,
    )
    body = json.loads(response_body.decode("utf-8"))
    text = body.get("text", "")
    print(f"  -> status {status}")
    print(f"  -> transcribed text: {text!r}")
    if not text.strip():
        raise RuntimeError("STT returned empty transcript for real audio fixture")
    print("  STT step PASSED")
    return text


def run_llm(text: str) -> str:
    prompt = text.strip()
    print("\nStep 2: POST to LLM (http://localhost:8002/v1/chat/completions) ...")
    print(f"  -> prompt: {prompt[:200]!r}{'...' if len(prompt) > 200 else ''}")
    status, _headers, response_body = _post_json(
        LLM_URL,
        {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Reply with exactly one short word.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": 5,
        },
        timeout=30,
    )
    body = json.loads(response_body.decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    print(f"  -> status {status}")
    print(f"  -> LLM response: {content!r}")
    if not content.strip():
        raise RuntimeError("LLM returned an empty assistant message")
    print("  LLM step PASSED")
    return content


def _minimal_tts_text(text: str) -> str:
    compact = " ".join(text.strip().split())
    if not compact:
        return "OK"
    return compact[:SMOKE_TTS_MAX_CHARS]


def run_tts(text: str) -> None:
    tts_text = _minimal_tts_text(text)
    print("\nStep 3: POST to TTS (http://localhost:8003/v1/audio/speech) ...")
    print(f"  -> minimal LLM-derived input text: {tts_text!r}")
    status, headers, response_body = _post_json(
        TTS_URL,
        {"input": tts_text},
        timeout=30,
    )
    ct = next((value for key, value in headers.items() if key.lower() == "content-type"), "")
    cl = len(response_body)
    print(f"  -> status {status}")
    print(f"  -> Content-Type: {ct}")
    print(f"  -> body size: {cl} bytes")
    if "audio/mpeg" not in ct:
        raise RuntimeError(f"Expected Content-Type audio/mpeg, got {ct}")
    if cl == 0:
        raise RuntimeError("TTS returned empty body")
    print("  TTS step PASSED")


def main() -> int:
    tmp_wav = Path("test_speech_fixture.wav")
    try:
        _wait_for_services()
        source_fixture = prepare_audio_fixture(tmp_wav)
        print(
            f"Using first {SMOKE_AUDIO_SECONDS:g}s from STT audio fixture: "
            f"{source_fixture.resolve()}\n"
        )

        text = run_stt(tmp_wav)
        llm_text = run_llm(text)
        run_tts(llm_text)

        print("\n" + "=" * 50)
        print("SMOKE TEST PASSED")
        print("=" * 50)
        return 0
    except urllib.error.HTTPError as exc:
        print(f"\nHTTP error {exc.code}: {_read_http_error(exc)}")
        return 1
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"\nConnection error: {exc}")
        print("Make sure the services are running: docker compose up -d")
        return 1
    except Exception as exc:
        print(f"\nSMOKE TEST FAILED: {exc}")
        return 1
    finally:
        if tmp_wav.exists():
            tmp_wav.unlink()


if __name__ == "__main__":
    sys.exit(main())
