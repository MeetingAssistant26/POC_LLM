import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Inject fake edge_tts module before importing main so tests run without
# needing network access to Microsoft Azure TTS edge endpoint.
fake_edge_tts = MagicMock()


class FakeCommunicate:
    """Fake edge_tts.Communicate that yields fake MP3 chunks."""

    def __init__(self, text, voice):
        self.text = text
        self.voice = voice

    async def stream(self):
        # Yield a fake audio chunk and a boundary chunk
        yield {"type": "audio", "data": b"\xff\xfb\x90"}  # Fake MP3 header-ish bytes
        yield {"type": "WordBoundary", "data": None}


fake_edge_tts.Communicate = FakeCommunicate
sys.modules["edge_tts"] = fake_edge_tts

from fastapi.testclient import TestClient
from main import app, _tts_state

client = TestClient(app)


class TestHealthz:
    def test_healthz_ready(self):
        _tts_state["ready"] = True
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == "tts"

    def test_healthz_not_ready(self):
        prev = _tts_state["ready"]
        _tts_state["ready"] = False
        response = client.get("/healthz")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["service"] == "tts"
        _tts_state["ready"] = prev


class TestSpeechEndpoint:
    def test_english_auto_detection(self):
        _tts_state["ready"] = True
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello world, this is a test."},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"\xff\xfb\x90"
        # Verify English voice was used
        assert FakeCommunicate.last_voice == "en-US-JennyNeural"

    def test_arabic_auto_detection(self):
        _tts_state["ready"] = True
        response = client.post(
            "/v1/audio/speech",
            json={"input": "مرحبا بالعالم، هذا اختبار."},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"\xff\xfb\x90"
        # Verify Arabic voice was used
        assert FakeCommunicate.last_voice == "ar-EG-ShakirNeural"

    def test_custom_voice_override(self):
        _tts_state["ready"] = True
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "Hello world.",
                "voice": "en-GB-SoniaNeural",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        # Verify custom voice was used
        assert FakeCommunicate.last_voice == "en-GB-SoniaNeural"

    def test_empty_input_returns_400(self):
        _tts_state["ready"] = True
        response = client.post(
            "/v1/audio/speech",
            json={"input": ""},
        )
        assert response.status_code == 400
        body = response.json()
        assert "input field is required" in body["detail"]

    def test_cleaning_removes_markdown(self):
        _tts_state["ready"] = True
        response = client.post(
            "/v1/audio/speech",
            json={"input": "**Bold** and *italic* text.\n- bullet point one\n- bullet point two"},
        )
        assert response.status_code == 200
        # Verify the text was cleaned before being passed to Communicate
        assert "**" not in FakeCommunicate.last_text
        assert "*" not in FakeCommunicate.last_text
        # The regex only strips leading "- " at start of lines (re.MULTILINE)
        assert "- bullet point one" not in FakeCommunicate.last_text
        assert "- bullet point two" not in FakeCommunicate.last_text

    def test_response_format_ignored(self):
        _tts_state["ready"] = True
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello.", "response_format": "wav"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"

    def test_speed_ignored(self):
        _tts_state["ready"] = True
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello.", "speed": 1.5},
        )
        assert response.status_code == 200


# Monkey-patch FakeCommunicate to capture constructor args
_original_init = FakeCommunicate.__init__


def _capturing_init(self, text, voice):
    self.text = text
    self.voice = voice
    FakeCommunicate.last_text = text
    FakeCommunicate.last_voice = voice


FakeCommunicate.__init__ = _capturing_init
FakeCommunicate.last_text = None
FakeCommunicate.last_voice = None
