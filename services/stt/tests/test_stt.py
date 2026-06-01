import os
import sys
import tempfile
from unittest.mock import MagicMock

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
from services.stt.main import app, _stt_state

client = TestClient(app)


@pytest.fixture(autouse=True)
def ready_stt_state():
    previous = _stt_state.copy()
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
