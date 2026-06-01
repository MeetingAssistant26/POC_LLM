import sys
from unittest.mock import MagicMock, patch

fake_edge_tts = MagicMock()


class FakeCommunicate:
    last_text = None
    last_voice = None

    def __init__(self, text, voice):
        self.text = text
        self.voice = voice
        FakeCommunicate.last_text = text
        FakeCommunicate.last_voice = voice

    async def stream(self):
        yield {"type": "audio", "data": b"\xff\xfb\x90"}
        yield {"type": "WordBoundary", "data": None}


fake_edge_tts.Communicate = FakeCommunicate
sys.modules["edge_tts"] = fake_edge_tts

fake_requests = MagicMock()
sys.modules["requests"] = fake_requests

from fastapi.testclient import TestClient
from services.tts.main import app, _tts_state

client = TestClient(app)


class TestHealthz:
    def test_healthz_ready(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        _tts_state["ready"] = True
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == "tts"
        assert body["provider"] == "elevenlabs"

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
    def test_elevenlabs_when_configured(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-123")
        fake_response = MagicMock(status_code=200, content=b"ID3fake-mp3")
        with patch("requests.post", return_value=fake_response) as post:
            response = client.post("/v1/audio/speech", json={"input": "Hello world."})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"ID3fake-mp3"
        assert "voice-123" in post.call_args.args[0]
        assert post.call_args.kwargs["headers"]["xi-api-key"] == "test-key"
        assert post.call_args.kwargs["json"]["model_id"] == "eleven_multilingual_v2"
        assert post.call_args.kwargs["params"] == {"output_format": "mp3_44100_128"}

    def test_custom_elevenlabs_voice_override(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        fake_response = MagicMock(status_code=200, content=b"ID3fake-mp3")
        with patch("requests.post", return_value=fake_response) as post:
            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello world.", "voice": "custom-voice"},
            )
        assert response.status_code == 200
        assert "custom-voice" in post.call_args.args[0]

    def test_elevenlabs_prefers_accessible_voice_when_configured_voice_is_not_listed(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        monkeypatch.setenv("ELEVENLABS_VOICE_ID", "blocked-voice")
        success_response = MagicMock(status_code=200, content=b"ID3fallback-mp3")
        voices_response = MagicMock(
            status_code=200,
            json=lambda: {"voices": [{"voice_id": "accessible-voice", "name": "Fallback"}]},
        )
        with patch("requests.post", return_value=success_response) as post, patch(
            "requests.get", return_value=voices_response
        ) as get:
            response = client.post("/v1/audio/speech", json={"input": "Hello world."})
        assert response.status_code == 200
        assert response.content == b"ID3fallback-mp3"
        assert "accessible-voice" in post.call_args_list[0].args[0]
        assert "blocked-voice" not in post.call_args_list[0].args[0]
        assert get.call_count == 1

    def test_elevenlabs_resolves_configured_voice_name_before_request(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        monkeypatch.setenv("ELEVENLABS_VOICE_NAME", "Named Voice")
        success_response = MagicMock(status_code=200, content=b"ID3named-mp3")
        voices_response = MagicMock(
            status_code=200,
            json=lambda: {"voices": [{"voice_id": "voice-id-for-name", "name": "Named Voice"}]},
        )
        with patch("requests.post", return_value=success_response) as post, patch(
            "requests.get", return_value=voices_response
        ):
            response = client.post("/v1/audio/speech", json={"input": "Hello world."})
        assert response.status_code == 200
        assert response.content == b"ID3named-mp3"
        assert "voice-id-for-name" in post.call_args_list[0].args[0]
        assert "Named%20Voice" not in post.call_args_list[0].args[0]

    def test_elevenlabs_explicit_voice_override_retries_accessible_voice_after_401(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        blocked_response = MagicMock(status_code=401, content=b"", json=lambda: {"detail": {"status": "voice_not_found"}})
        success_response = MagicMock(status_code=200, content=b"ID3fallback-mp3")
        voices_response = MagicMock(
            status_code=200,
            json=lambda: {"voices": [{"voice_id": "accessible-voice", "name": "Fallback"}]},
        )
        with patch("requests.post", side_effect=[blocked_response, success_response]) as post, patch(
            "requests.get", return_value=voices_response
        ):
            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello world.", "voice": "blocked-voice"},
            )
        assert response.status_code == 200
        assert response.content == b"ID3fallback-mp3"
        assert "blocked-voice" in post.call_args_list[0].args[0]
        assert "accessible-voice" in post.call_args_list[1].args[0]

    def test_elevenlabs_failure_returns_502(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        fake_response = MagicMock(status_code=403, content=b"", text="forbidden", json=lambda: {})
        with patch("requests.post", return_value=fake_response):
            response = client.post("/v1/audio/speech", json={"input": "Hello world."})
        assert response.status_code == 502
        assert "ElevenLabs TTS failed" in response.json()["detail"]

    def test_elevenlabs_quota_401_does_not_retry_voices(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        quota_response = MagicMock(
            status_code=401,
            content=b"",
            json=lambda: {"detail": {"status": "quota_exceeded", "message": "0 credits remaining"}},
        )
        voices_response = MagicMock(
            status_code=200,
            json=lambda: {"voices": [{"voice_id": "accessible-voice", "name": "Fallback"}]},
        )
        with patch("requests.post", return_value=quota_response) as post, patch(
            "requests.get", return_value=voices_response
        ):
            response = client.post("/v1/audio/speech", json={"input": "Hello world."})
        assert response.status_code == 502
        assert post.call_count == 1
        assert "quota_exceeded" in response.json()["detail"]

    def test_edge_fallback_when_elevenlabs_unconfigured(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        response = client.post("/v1/audio/speech", json={"input": "Hello world."})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"\xff\xfb\x90"
        assert FakeCommunicate.last_voice == "en-US-JennyNeural"

    def test_arabic_edge_fallback_detection(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        response = client.post("/v1/audio/speech", json={"input": "مرحبا بالعالم، هذا اختبار."})
        assert response.status_code == 200
        assert FakeCommunicate.last_voice == "ar-EG-ShakirNeural"

    def test_empty_input_returns_400(self):
        _tts_state["ready"] = True
        response = client.post("/v1/audio/speech", json={"input": ""})
        assert response.status_code == 400
        assert "input field is required" in response.json()["detail"]

    def test_cleaning_removes_markdown(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        response = client.post(
            "/v1/audio/speech",
            json={"input": "**Bold** and *italic* text.\n- bullet point one\n- bullet point two"},
        )
        assert response.status_code == 200
        assert "**" not in FakeCommunicate.last_text
        assert "*" not in FakeCommunicate.last_text
        assert "- bullet point one" not in FakeCommunicate.last_text
        assert "- bullet point two" not in FakeCommunicate.last_text

    def test_pcm_response_format_uses_elevenlabs_pcm_output(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-123")
        fake_response = MagicMock(status_code=200, content=b"\x01\x02raw-pcm")
        with patch("requests.post", return_value=fake_response) as post:
            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello.", "response_format": "pcm"},
            )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.content == b"\x01\x02raw-pcm"
        assert post.call_args.kwargs["params"] == {"output_format": "pcm_24000"}
        assert post.call_args.kwargs["headers"]["Accept"] == "application/octet-stream"

    def test_unsupported_response_format_returns_400(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        response = client.post("/v1/audio/speech", json={"input": "Hello.", "response_format": "wav"})
        assert response.status_code == 400
        assert "response_format" in response.json()["detail"]

    def test_pcm_response_format_without_elevenlabs_returns_501(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        response = client.post("/v1/audio/speech", json={"input": "Hello.", "response_format": "pcm"})
        assert response.status_code == 501
        assert "requires ElevenLabs" in response.json()["detail"]

    def test_speed_ignored(self, monkeypatch):
        _tts_state["ready"] = True
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        response = client.post("/v1/audio/speech", json={"input": "Hello.", "speed": 1.5})
        assert response.status_code == 200
