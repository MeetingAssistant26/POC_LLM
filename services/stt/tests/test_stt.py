import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest
import soundfile as sf

# Inject fake heavy modules before importing main so tests run without
# installing torch + whisperx.
fake_whisperx = MagicMock()
fake_torch = MagicMock()
fake_torch.cuda.is_available.return_value = False

fake_model = MagicMock()
fake_whisperx.load_model.return_value = fake_model
fake_whisperx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

fake_transcribe_result = {
    "language": "en",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello world"}
    ],
}
fake_model.transcribe.return_value = fake_transcribe_result

fake_align_model = MagicMock()
fake_metadata = MagicMock()
fake_whisperx.load_align_model.return_value = (fake_align_model, fake_metadata)
fake_whisperx.align.return_value = {
    "language": "en",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello world", "speaker": "SPEAKER_00"}
    ],
}

fake_diarize_pipeline = MagicMock()
fake_diarize_pipeline.return_value = MagicMock()
fake_diarize_module = MagicMock()
fake_diarize_module.DiarizationPipeline.return_value = fake_diarize_pipeline
fake_whisperx.diarize = fake_diarize_module

fake_whisperx.assign_word_speakers.return_value = {
    "language": "en",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello world", "speaker": "SPEAKER_00"}
    ],
}

sys.modules["whisperx"] = fake_whisperx
sys.modules["whisperx.diarize"] = fake_diarize_module
sys.modules["torch"] = fake_torch

from fastapi.testclient import TestClient
from services.stt.main import (
    _parse_chat_audio_transcription,
    _stt_state,
    _upstream_request_format,
    app,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def ready_stt_state(monkeypatch):
    previous = _stt_state.copy()
    for name in (
        "STT_PROVIDER",
        "STT_UPSTREAM_BASE_URL",
        "STT_UPSTREAM_API_KEY",
        "STT_UPSTREAM_MODEL",
        "STT_UPSTREAM_REQUEST_FORMAT",
        "STT_UPSTREAM_LANGUAGE",
    ):
        monkeypatch.delenv(name, raising=False)
    _stt_state["model"] = fake_model
    _stt_state["device"] = "cpu"
    _stt_state["compute_type"] = "int8"
    _stt_state["hf_token"] = None
    fake_diarize_module.DiarizationPipeline.reset_mock(return_value=True, side_effect=True)
    fake_diarize_module.DiarizationPipeline.return_value = fake_diarize_pipeline
    fake_whisperx.align.reset_mock()
    fake_whisperx.align.return_value = {
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello world", "speaker": "SPEAKER_00"}
        ],
    }
    fake_whisperx.assign_word_speakers.reset_mock()
    fake_whisperx.assign_word_speakers.return_value = {
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello world", "speaker": "SPEAKER_00"}
        ],
    }
    yield
    _stt_state.update(previous)


def _generate_test_wav() -> str:
    """Generate a tiny 1-second 16 kHz mono wav file in a temp path."""
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    data = np.sin(2 * np.pi * 440 * t) * 0.1
    data = data.astype(np.float32)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, data, sample_rate)
    return path


class TestHealthz:
    def test_healthz_ready(self):
        _stt_state["model"] = fake_model
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_healthz_not_ready(self):
        prev = _stt_state["model"]
        _stt_state["model"] = None
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        _stt_state["model"] = prev

    def test_readyz_not_ready(self):
        prev = _stt_state["model"]
        _stt_state["model"] = None
        response = client.get("/readyz")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["model_loaded"] is False
        _stt_state["model"] = prev

    def test_readyz_ready(self):
        prev = _stt_state["model"]
        _stt_state["model"] = fake_model
        response = client.get("/readyz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        _stt_state["model"] = prev

    def test_readyz_openai_compatible_missing_config(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.delenv("STT_UPSTREAM_BASE_URL", raising=False)
        monkeypatch.delenv("STT_UPSTREAM_API_KEY", raising=False)

        response = client.get("/readyz")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["provider"] == "openai-compatible"
        assert body["missing"] == ["STT_UPSTREAM_BASE_URL", "STT_UPSTREAM_API_KEY"]

    def test_readyz_openai_compatible_ready_without_local_model(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://stt-upstream.example/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        _stt_state["model"] = None

        response = client.get("/readyz")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["provider"] == "openai-compatible"
        assert body["upstream_configured"] is True

    def test_openrouter_auto_uses_json_for_transcription_voxtral_models(self, monkeypatch):
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "mistralai/voxtral-mini-transcribe")
        monkeypatch.setenv("STT_UPSTREAM_REQUEST_FORMAT", "auto")

        assert _upstream_request_format() == "openrouter-json"

    def test_openrouter_auto_uses_chat_audio_for_non_transcription_voxtral_models(self, monkeypatch):
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "mistralai/voxtral-small-24b-2507")
        monkeypatch.setenv("STT_UPSTREAM_REQUEST_FORMAT", "auto")

        assert _upstream_request_format() == "openrouter-chat-audio"

    def test_openrouter_auto_uses_chat_audio_for_gemini_models(self, monkeypatch):
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "google/gemini-3.1-flash-lite")
        monkeypatch.setenv("STT_UPSTREAM_REQUEST_FORMAT", "auto")

        assert _upstream_request_format() == "openrouter-chat-audio"

    def test_readyz_openai_compatible_exposes_gemini_runtime_config(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "google/gemini-3.1-flash-lite")
        monkeypatch.setenv("STT_UPSTREAM_REQUEST_FORMAT", "openrouter-chat-audio")
        _stt_state["model"] = None

        response = client.get("/readyz")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["provider"] == "openai-compatible"
        assert body["upstream_configured"] is True
        assert body["upstream_model"] == "google/gemini-3.1-flash-lite"
        assert body["request_format"] == "openrouter-chat-audio"
        assert body["upstream_endpoint"] == "/chat/completions"
        assert body["upstream_base_url"] == "https://openrouter.ai/api/v1"

    @pytest.mark.parametrize("request_format", ["openrouter-json", "openai-multipart"])
    def test_gemini_misconfigured_request_format_fails_readiness(self, monkeypatch, request_format):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "google/gemini-3.1-flash-lite")
        monkeypatch.setenv("STT_UPSTREAM_REQUEST_FORMAT", request_format)

        response = client.get("/readyz")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["provider"] == "openai-compatible"
        assert "openrouter-chat-audio" in body["error"]
        assert "google/gemini" in body["error"]


class TestTranscriptions:
    def test_json_response(self):
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"response_format": "json"},
                )
            assert response.status_code == 200
            body = response.json()
            assert "text" in body
            assert body["text"] != ""
            assert "segments" in body
            assert len(body["segments"]) > 0
        finally:
            os.unlink(wav_path)

    def test_openai_compatible_proxy_forwards_audio_and_configured_model(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://stt-upstream.example/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "upstream-whisper-model")
        _stt_state["model"] = None
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured["timeout"] = kwargs.get("timeout")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def post(self, url, headers, data, files):
                captured["url"] = url
                captured["headers"] = headers
                captured["data"] = data
                captured["files"] = files
                return httpx.Response(
                    200,
                    json={"text": "proxied transcript", "segments": [], "language": "en"},
                    headers={"content-type": "application/json"},
                )

        monkeypatch.setattr("services.stt.main.httpx.Client", FakeClient)
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={
                        "model": "request-whisper-model",
                        "language": "en",
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "segment",
                    },
                )

            assert response.status_code == 200
            assert response.json()["text"] == "proxied transcript"
            assert captured["url"] == "https://stt-upstream.example/v1/audio/transcriptions"
            assert captured["headers"] == {"Authorization": "Bearer test-upstream-key"}
            assert captured["data"]["model"] == "upstream-whisper-model"
            assert captured["data"]["language"] == "en"
            assert captured["data"]["response_format"] == "verbose_json"
            assert captured["data"]["timestamp_granularities[]"] == "segment"
            assert captured["files"]["file"][0] == "test.wav"
            assert captured["files"]["file"][2] == "audio/wav"
            assert captured["files"]["file"][1]
        finally:
            os.unlink(wav_path)

    def test_openai_compatible_proxy_defaults_response_format_and_uses_request_model(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://stt-upstream.example/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.delenv("STT_UPSTREAM_MODEL", raising=False)
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def post(self, url, headers, data, files):
                captured["data"] = data
                return httpx.Response(
                    200,
                    json={"text": "proxied transcript"},
                    headers={"content-type": "application/json"},
                )

        monkeypatch.setattr("services.stt.main.httpx.Client", FakeClient)
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"model": "request-whisper-model"},
                )

            assert response.status_code == 200
            assert captured["data"]["model"] == "request-whisper-model"
            assert captured["data"]["response_format"] == "verbose_json"
        finally:
            os.unlink(wav_path)

    def test_openai_compatible_proxy_supports_openrouter_json_audio(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "openrouter-transcription-model")
        monkeypatch.setenv("STT_UPSTREAM_REQUEST_FORMAT", "openrouter-json")
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured["timeout"] = kwargs.get("timeout")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def post(self, url, headers, json):
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return httpx.Response(
                    200,
                    json={"text": "openrouter transcript", "usage": {"seconds": 1.0}},
                    headers={"content-type": "application/json"},
                )

        monkeypatch.setattr("services.stt.main.httpx.Client", FakeClient)
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"model": "ignored-by-config", "language": "en"},
                )

            assert response.status_code == 200
            assert response.json()["text"] == "openrouter transcript"
            assert captured["url"] == "https://openrouter.ai/api/v1/audio/transcriptions"
            assert captured["headers"]["Authorization"] == "Bearer test-upstream-key"
            assert captured["headers"]["Content-Type"] == "application/json"
            assert captured["json"]["model"] == "openrouter-transcription-model"
            assert captured["json"]["language"] == "en"
            assert captured["json"]["input_audio"]["format"] == "wav"
            assert captured["json"]["input_audio"]["data"]
        finally:
            os.unlink(wav_path)

    def test_openai_compatible_proxy_can_omit_openrouter_json_language_for_auto_detection(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "microsoft/mai-transcribe-1.5")
        monkeypatch.setenv("STT_UPSTREAM_REQUEST_FORMAT", "openrouter-json")
        monkeypatch.setenv("STT_UPSTREAM_LANGUAGE", "auto")
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def post(self, url, headers, json):
                captured["json"] = json
                return httpx.Response(
                    200,
                    json={"text": "auto-detected transcript"},
                    headers={"content-type": "application/json"},
                )

        monkeypatch.setattr("services.stt.main.httpx.Client", FakeClient)
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"model": "ignored-by-config", "language": "en"},
                )

            assert response.status_code == 200
            assert "language" not in captured["json"]
        finally:
            os.unlink(wav_path)

    def test_openai_compatible_proxy_can_override_openrouter_json_language(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "microsoft/mai-transcribe-1.5")
        monkeypatch.setenv("STT_UPSTREAM_REQUEST_FORMAT", "openrouter-json")
        monkeypatch.setenv("STT_UPSTREAM_LANGUAGE", "ar")
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def post(self, url, headers, json):
                captured["json"] = json
                return httpx.Response(
                    200,
                    json={"text": "arabic transcript"},
                    headers={"content-type": "application/json"},
                )

        monkeypatch.setattr("services.stt.main.httpx.Client", FakeClient)
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"model": "ignored-by-config", "language": "en"},
                )

            assert response.status_code == 200
            assert captured["json"]["language"] == "ar"
        finally:
            os.unlink(wav_path)

    def test_openai_compatible_proxy_supports_openrouter_chat_audio_for_gemini(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "google/gemini-3.1-flash-lite")
        monkeypatch.setenv("STT_UPSTREAM_REQUEST_FORMAT", "openrouter-chat-audio")
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured["timeout"] = kwargs.get("timeout")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def post(self, url, headers, json):
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return httpx.Response(
                    200,
                    json={
                        "model": "google/gemini-3.1-flash-lite",
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        '{"segments":[{"t":0.5,"text":"gemini transcript"}]}'
                                    )
                                }
                            }
                        ],
                        "usage": {"total_tokens": 8},
                    },
                    headers={"content-type": "application/json"},
                )

        monkeypatch.setattr("services.stt.main.httpx.Client", FakeClient)
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"model": "ignored-by-config"},
                )

            assert response.status_code == 200
            body = response.json()
            assert body["text"] == "gemini transcript"
            assert body["segments"] == [
                {"id": 0, "start": 0.5, "end": 0.5, "text": "gemini transcript"}
            ]
            assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
            assert captured["json"]["model"] == "google/gemini-3.1-flash-lite"
            content = captured["json"]["messages"][0]["content"]
            assert content[0]["type"] == "text"
            assert "Schema" in content[0]["text"]
            assert content[1]["type"] == "input_audio"
            assert content[1]["input_audio"]["format"] == "wav"
            assert content[1]["input_audio"]["data"]
        finally:
            os.unlink(wav_path)

    def test_openai_compatible_proxy_supports_openrouter_chat_audio_for_voxtral(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "mistralai/voxtral-small-24b-2507")
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured["timeout"] = kwargs.get("timeout")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def post(self, url, headers, json):
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": '{"segments":[{"t":1.25,"text":"chat audio transcript"}]}'
                                }
                            }
                        ],
                        "usage": {"total_tokens": 12},
                    },
                    headers={"content-type": "application/json"},
                )

        monkeypatch.setattr("services.stt.main.httpx.Client", FakeClient)
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"model": "ignored-by-config"},
                )

            assert response.status_code == 200
            body = response.json()
            assert body["text"] == "chat audio transcript"
            assert body["segments"] == [{"id": 0, "start": 1.25, "end": 1.25, "text": "chat audio transcript"}]
            assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
            assert captured["json"]["model"] == "mistralai/voxtral-small-24b-2507"
            content = captured["json"]["messages"][0]["content"]
            assert content[0]["type"] == "text"
            assert "Schema" in content[0]["text"]
            assert content[1]["type"] == "input_audio"
            assert content[1]["input_audio"]["format"] == "wav"
            assert content[1]["input_audio"]["data"]
        finally:
            os.unlink(wav_path)

    def test_parse_chat_audio_malformed_structured_json_does_not_leak_raw(self):
        malformed = (
            '{"segments":[{"t":13.1,"text":"Hello assistant"},'
            '{"t":14.2,"text":"follow up phrase"}'
        )
        body = _parse_chat_audio_transcription(
            {
                "choices": [{"message": {"content": malformed}}],
                "usage": {"total_tokens": 3},
            }
        )

        assert body["text"] == "Hello assistant follow up phrase"
        assert body["segments"] == [
            {"id": 0, "start": 0.0, "end": 0.0, "text": "Hello assistant"},
            {"id": 1, "start": 0.0, "end": 0.0, "text": "follow up phrase"},
        ]
        assert '{"segments"' not in body["text"]
        for segment in body["segments"]:
            assert '{"segments"' not in segment["text"]

    def test_parse_chat_audio_fenced_valid_json_is_parsed(self):
        fenced = (
            '```json\n'
            '{"segments":[{"t":2.5,"text":"fenced transcript"}]}\n'
            "```"
        )
        body = _parse_chat_audio_transcription(
            {"choices": [{"message": {"content": fenced}}]}
        )

        assert body["text"] == "fenced transcript"
        assert body["segments"] == [
            {"id": 0, "start": 2.5, "end": 2.5, "text": "fenced transcript"}
        ]

    def test_parse_chat_audio_plain_text_fallback_still_works(self):
        body = _parse_chat_audio_transcription(
            {"choices": [{"message": {"content": "plain spoken transcript"}}]}
        )

        assert body["text"] == "plain spoken transcript"
        assert body["segments"] == [
            {"id": 0, "start": 0.0, "end": 0.0, "text": "plain spoken transcript"}
        ]

    def test_parse_chat_audio_nested_root_text_valid_json(self):
        content = '{"text":"{\\"segments\\":[{\\"text\\":\\"hello\\"}]}"}'
        body = _parse_chat_audio_transcription(
            {"choices": [{"message": {"content": content}}]}
        )

        assert body["text"] == "hello"
        assert body["segments"] == [
            {"id": 0, "start": 0.0, "end": 0.0, "text": "hello"}
        ]
        assert '{"segments"' not in body["text"]
        for segment in body["segments"]:
            assert '{"segments"' not in segment["text"]

    def test_parse_chat_audio_nested_root_text_malformed_json(self):
        content = '{"text":"{\\"segments\\":[{\\"text\\":\\"hello\\"}"}'
        body = _parse_chat_audio_transcription(
            {"choices": [{"message": {"content": content}}]}
        )

        assert body["text"] == "hello"
        assert body["segments"] == [
            {"id": 0, "start": 0.0, "end": 0.0, "text": "hello"}
        ]
        assert '{"segments"' not in body["text"]
        for segment in body["segments"]:
            assert '{"segments"' not in segment["text"]

    def test_openai_compatible_proxy_suppresses_common_empty_audio_hallucination(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "mistralai/voxtral-small-24b-2507")

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def post(self, url, headers, json):
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        "Hello, everyone. Welcome to the podcast. "
                                        "Today, we have a special guest. Let's get started."
                                    )
                                }
                            }
                        ],
                        "usage": {"total_tokens": 12},
                    },
                    headers={"content-type": "application/json"},
                )

        monkeypatch.setattr("services.stt.main.httpx.Client", FakeClient)
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"model": "ignored-by-config"},
                )

            assert response.status_code == 200
            assert response.json()["text"] == ""
            assert response.json()["segments"] == []
        finally:
            os.unlink(wav_path)



    def test_gemini_trace_step_includes_request_format_and_model(self, monkeypatch):
        monkeypatch.setenv("STT_PROVIDER", "openai-compatible")
        monkeypatch.setenv("STT_UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("STT_UPSTREAM_API_KEY", "test-upstream-key")
        monkeypatch.setenv("STT_UPSTREAM_MODEL", "google/gemini-3.1-flash-lite")
        monkeypatch.setenv("STT_UPSTREAM_REQUEST_FORMAT", "openrouter-chat-audio")

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def post(self, url, headers, json):
                return httpx.Response(
                    200,
                    json={
                        "model": "google/gemini-3.1-flash-lite",
                        "choices": [
                            {
                                "message": {
                                    "content": '{"segments":[{"t":0,"text":"trace transcript"}]}'
                                }
                            }
                        ],
                    },
                    headers={"content-type": "application/json"},
                )

        monkeypatch.setattr("services.stt.main.httpx.Client", FakeClient)
        wav_path = _generate_test_wav()
        headers = {
            "X-AI-Trace-Enabled": "true",
            "X-AI-Trace-Session-Id": "session-gemini",
            "X-AI-Trace-Turn-Id": "turn-gemini",
            "X-AI-Trace-Sequence-Base": "1",
            "X-AI-Trace-Meeting-Id": "meeting-gemini",
            "X-AI-Trace-Organization-Id": "org-gemini",
            "X-AI-Trace-Backend-Url": "http://api:8080",
            "X-AI-Trace-Agent-Token": "token-secret",
            "X-AI-Trace-Persist-Payloads": "true",
        }
        try:
            with patch("services.stt.main.post_trace_event", new_callable=AsyncMock) as post_trace, open(
                wav_path, "rb"
            ) as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"model": "ignored-by-config"},
                    headers=headers,
                )
            assert response.status_code == 200
            post_trace.assert_awaited_once()
            step = post_trace.call_args.kwargs["step"]
            assert step["requestFormat"] == "openrouter-chat-audio"
            assert step["model"] == "google/gemini-3.1-flash-lite"
            assert step["upstreamResponseModel"] == "google/gemini-3.1-flash-lite"
            assert "voxtral" not in str(step).lower()
        finally:
            os.unlink(wav_path)

    def test_trace_headers_emit_stt_completed(self):
        wav_path = _generate_test_wav()
        headers = {
            "X-AI-Trace-Enabled": "true",
            "X-AI-Trace-Session-Id": "session-1",
            "X-AI-Trace-Turn-Id": "turn-1",
            "X-AI-Trace-Sequence-Base": "10",
            "X-AI-Trace-Meeting-Id": "meeting-1",
            "X-AI-Trace-Organization-Id": "org-1",
            "X-AI-Trace-Backend-Url": "http://api:8080",
            "X-AI-Trace-Agent-Token": "token-secret",
            "X-AI-Trace-Persist-Payloads": "true",
        }
        try:
            with patch("services.stt.main.post_trace_event", new_callable=AsyncMock) as post_trace, open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"response_format": "json"},
                    headers=headers,
                )
            assert response.status_code == 200
            post_trace.assert_awaited_once()
            args, kwargs = post_trace.call_args
            assert args[1] == "stt_completed"
            assert kwargs["step"]["type"] == "stt"
            assert kwargs["step"]["provider"] == "WhisperX"
            assert kwargs["step"]["model"] == "medium"
            assert kwargs["step"]["text"] == "Hello world"
        finally:
            os.unlink(wav_path)

    def test_text_response(self):
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"response_format": "text"},
                )
            assert response.status_code == 200
            body = response.json()
            assert "text" in body
            assert body["text"] != ""
        finally:
            os.unlink(wav_path)

    def test_srt_response(self):
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"response_format": "srt"},
                )
            assert response.status_code == 200
            body = response.json()
            assert "text" in body
            assert "-->" in body["text"]
        finally:
            os.unlink(wav_path)

    def test_vtt_response(self):
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"response_format": "vtt"},
                )
            assert response.status_code == 200
            body = response.json()
            assert "text" in body
            assert "WEBVTT" in body["text"]
        finally:
            os.unlink(wav_path)

    def test_speaker_included_when_diarized(self):
        _stt_state["hf_token"] = "hf_test_token"
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"response_format": "verbose_json"},
                )
            assert response.status_code == 200
            body = response.json()
            segments = body["segments"]
            fake_diarize_module.DiarizationPipeline.assert_called_once_with(
                use_auth_token="hf_test_token",
                device="cpu",
            )
            assert any("speaker" in seg for seg in segments)
        finally:
            os.unlink(wav_path)

    def test_transcription_survives_incompatible_diarization_constructor(self):
        _stt_state["hf_token"] = "hf_test_token"
        fake_diarize_module.DiarizationPipeline.side_effect = TypeError("unexpected keyword argument")
        fake_whisperx.align.return_value = {
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello world"}
            ],
        }
        wav_path = _generate_test_wav()
        try:
            with open(wav_path, "rb") as f:
                response = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"response_format": "verbose_json"},
                )
            assert response.status_code == 200
            body = response.json()
            assert body["text"] == "Hello world"
            assert body["segments"] == [
                {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello world"}
            ]
            assert fake_diarize_module.DiarizationPipeline.call_count == 3
            fake_whisperx.assign_word_speakers.assert_not_called()
        finally:
            os.unlink(wav_path)
