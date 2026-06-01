import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Inject fake groq module before importing main so tests run without
# needing a real GROQ_API_KEY or network access.
fake_groq = MagicMock()

fake_message = MagicMock()
fake_message.role = "assistant"
fake_message.content = "Hello, I am a helpful assistant."

fake_choice = MagicMock()
fake_choice.index = 0
fake_choice.message = fake_message
fake_choice.finish_reason = "stop"

fake_usage = MagicMock()
fake_usage.prompt_tokens = 10
fake_usage.completion_tokens = 20
fake_usage.total_tokens = 30

fake_response = MagicMock()
fake_response.choices = [fake_choice]
fake_response.usage = fake_usage

fake_client_instance = MagicMock()
fake_client_instance.chat.completions.create.return_value = fake_response

fake_groq.Groq.return_value = fake_client_instance
sys.modules["groq"] = fake_groq

# Also mock dotenv so load_dotenv is a no-op
sys.modules["dotenv"] = MagicMock()
sys.modules["python_dotenv"] = MagicMock()

# Ensure GROQ_API_KEY is set for lifespan initialization
os.environ["GROQ_API_KEY"] = "fake-groq-api-key-for-tests"

from fastapi.testclient import TestClient
from main import app, _llm_state

client = TestClient(app)


class TestHealthz:
    def test_healthz_ready(self):
        _llm_state["client"] = fake_client_instance
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["client_initialized"] is True

    def test_healthz_not_ready(self):
        prev = _llm_state["client"]
        _llm_state["client"] = None
        response = client.get("/healthz")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["client_initialized"] is False
        _llm_state["client"] = prev


class TestChatCompletions:
    def test_non_streaming_json(self):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = fake_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert "id" in body
        assert body["model"] == "llama-3.3-70b-versatile"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "Hello, I am a helpful assistant."
        assert body["choices"][0]["finish_reason"] == "stop"
        assert "usage" in body
        assert body["usage"]["prompt_tokens"] == 10
        assert body["usage"]["completion_tokens"] == 20
        assert body["usage"]["total_tokens"] == 30

    def test_default_model_when_missing(self):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = fake_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        # Verify Groq was called with default model
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    def test_default_model_for_openai_alias(self):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = fake_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    def test_optional_params_forwarded(self):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = fake_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "temperature": 0.5,
                "max_tokens": 100,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
                "stop": ["END"],
            },
        )
        assert response.status_code == 200
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["frequency_penalty"] == 0.1
        assert call_kwargs["presence_penalty"] == 0.2
        assert call_kwargs["stop"] == ["END"]

    def test_streaming_sse(self):
        _llm_state["client"] = fake_client_instance

        # Build a fake stream that yields two content chunks + a finish chunk
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.role = None
        chunk1.choices[0].finish_reason = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " world"
        chunk2.choices[0].delta.role = None
        chunk2.choices[0].finish_reason = None

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta = MagicMock()
        chunk3.choices[0].delta.content = None
        chunk3.choices[0].delta.role = None
        chunk3.choices[0].finish_reason = "stop"

        fake_stream = [chunk1, chunk2, chunk3]
        fake_client_instance.chat.completions.create.return_value = fake_stream

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        raw_body = response.content.decode("utf-8")
        lines = raw_body.strip().split("\n\n")

        # Should have: role chunk + 2 content chunks + finish chunk + [DONE]
        assert len(lines) >= 4

        # First line should be role delta
        first_data = lines[0].replace("data: ", "")
        first_json = json.loads(first_data)
        assert first_json["object"] == "chat.completion.chunk"
        assert first_json["choices"][0]["delta"]["role"] == "assistant"

        # Collect content
        contents = []
        for line in lines[1:]:
            if line == "data: [DONE]":
                break
            data = line.replace("data: ", "")
            parsed = json.loads(data)
            delta = parsed["choices"][0]["delta"]
            if "content" in delta:
                contents.append(delta["content"])
            if parsed["choices"][0]["finish_reason"]:
                assert parsed["choices"][0]["finish_reason"] == "stop"

        assert "".join(contents) == "Hello world"
        assert "data: [DONE]" in raw_body

    def test_client_not_initialized_returns_503(self):
        prev = _llm_state["client"]
        _llm_state["client"] = None
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 503
        body = response.json()
        assert "Groq client is not initialized" in body["detail"]
        _llm_state["client"] = prev
