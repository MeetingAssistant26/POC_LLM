import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Tests inject a fake OpenAI-compatible client directly into _llm_state so
# they run without a real LLM_API_KEY or network access.

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

fake_openai = MagicMock()
fake_openai.OpenAI.return_value = fake_client_instance
sys.modules["openai"] = fake_openai


# Also mock dotenv so load_dotenv is a no-op
sys.modules["dotenv"] = MagicMock()
sys.modules["python_dotenv"] = MagicMock()

# Ensure LLM settings are set for lifespan initialization
os.environ["LLM_BASE_URL"] = "https://example.test/v1"
os.environ["LLM_API_KEY"] = "fake-llm-api-key-for-tests"
os.environ["LLM_MODEL"] = "llama-3.3-70b-versatile"

from fastapi.testclient import TestClient
from services.llm.main import app, _llm_state

client = TestClient(app)


def trace_headers():
    return {
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


def stream_chunk(
    content=None,
    finish_reason=None,
    usage=None,
    choices=None,
    reasoning_content=None,
):
    if choices is None:
        delta_kwargs = {"content": content, "role": None}
        if reasoning_content is not None:
            delta_kwargs["reasoning_content"] = reasoning_content
        choices = [
            SimpleNamespace(
                delta=SimpleNamespace(**delta_kwargs),
                finish_reason=finish_reason,
            )
        ]
    return SimpleNamespace(choices=choices, usage=usage)


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


def parse_llm_trace_lines(capsys):
    captured = capsys.readouterr()
    entries = []
    for line in captured.out.splitlines():
        marker = "[LLM_TRACE] "
        if marker not in line:
            continue
        payload = json.loads(line.split(marker, 1)[1])
        entries.append(payload)
    return entries


SECRET_LIKE_KEYS = {
    "api_key",
    "apiKey",
    "authorization",
    "messages",
    "prompt",
    "content",
    "text",
    "requestPayload",
    "responsePayload",
    "agent_token",
    "agentToken",
    "reasoning_content",
    "reasoningContent",
}

RAW_TRACE_ID_VALUES = {
    "session-1",
    "turn-1",
    "meeting-1",
    "org-1",
    "token-secret",
    "http://api:8080",
}


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



    def test_trace_headers_emit_llm_completed(self):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = fake_response
        with patch("services.llm.main.post_trace_event", new_callable=AsyncMock) as post_trace:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "stream": False,
                },
                headers=trace_headers(),
            )
        assert response.status_code == 200
        post_trace.assert_awaited_once()
        args, kwargs = post_trace.call_args
        assert args[1] == "llm_completed"
        assert kwargs["step"]["provider"] == "OpenAI-compatible"
        assert kwargs["step"]["promptTokens"] == 10
        assert kwargs["step"]["completionTokens"] == 20
        assert kwargs["step"]["totalTokens"] == 30
        assert kwargs["step"]["text"] == "Hello, I am a helpful assistant."
        assert kwargs["step"]["requestPayload"]["messages"] == [{"role": "user", "content": "Say hello"}]

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
        # Verify upstream was called with configured default model
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

    def test_max_completion_tokens_forwarded_when_max_tokens_absent(self):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = fake_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "max_completion_tokens": 300,
            },
        )
        assert response.status_code == 200
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 300

    def test_camel_case_max_completion_tokens_alias_is_forwarded(self):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = fake_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "maxCompletionTokens": 512,
            },
        )
        assert response.status_code == 200
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 512

    def test_max_tokens_takes_precedence_over_max_completion_tokens(self):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = fake_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "max_tokens": 120,
                "max_completion_tokens": 300,
            },
        )
        assert response.status_code == 200
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 120

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

    def test_streaming_trace_captures_usage_final_text_and_request_payload(self):
        _llm_state["client"] = fake_client_instance
        usage = {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
        fake_stream = [
            stream_chunk(content="Hello"),
            stream_chunk(content=" world"),
            stream_chunk(content=None, finish_reason="stop"),
            stream_chunk(usage=usage, choices=[]),
        ]
        fake_client_instance.chat.completions.create.return_value = fake_stream
        request_body = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
        }

        with patch("services.llm.main.post_trace_event", new_callable=AsyncMock) as post_trace:
            response = client.post(
                "/v1/chat/completions",
                json=request_body,
                headers=trace_headers(),
            )

        assert response.status_code == 200
        raw_body = response.content.decode("utf-8")
        assert "Hello" in raw_body
        assert " world" in raw_body
        assert f'"usage": {json.dumps(usage)}' in raw_body

        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert "stream_options" not in call_kwargs

        post_trace.assert_awaited_once()
        args, kwargs = post_trace.call_args
        assert args[1] == "llm_completed"
        step = kwargs["step"]
        assert step["provider"] == "OpenAI-compatible"
        assert step["promptTokens"] == 11
        assert step["completionTokens"] == 5
        assert step["totalTokens"] == 16
        assert step["text"] == "Hello world"
        assert step["requestPayload"] == request_body
        assert step["requestPayload"]["messages"] == [
            {"role": "user", "content": "Say hello"}
        ]
        assert step["responsePayload"]["content"] == "Hello world"
        assert step["responsePayload"]["usage"] == usage

    @pytest.mark.parametrize("stream_options_key", ["stream_options", "streamOptions"])
    def test_streaming_accepts_but_does_not_forward_caller_stream_options(
        self, stream_options_key
    ):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = [
            stream_chunk(content=None, finish_reason="stop")
        ]
        supplied_stream_options = {"include_usage": False}

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
                stream_options_key: supplied_stream_options,
            },
        )

        assert response.status_code == 200
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert "stream_options" not in call_kwargs

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
        assert "LLM client is not initialized" in body["detail"]
        _llm_state["client"] = prev

    def test_non_streaming_emits_llm_trace_lines(self, capsys):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = fake_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": False,
                "max_tokens": 100,
            },
            headers=trace_headers(),
        )
        assert response.status_code == 200

        traces = parse_llm_trace_lines(capsys)
        assert len(traces) == 2
        started, completed = traces

        assert started["event"] == "request_started"
        assert started["stream"] is False
        assert started["messageCount"] == 1
        assert started["maxTokensSource"] == "max_tokens"
        assert started["forwardedMaxTokens"] == 100
        assert started["traceEnabled"] is True
        assert started["traceSessionIdPresent"] is True
        assert started["traceAgentTokenPresent"] is True
        assert isinstance(started["requestId"], str)
        assert started["requestId"].startswith("chatcmpl-")
        assert isinstance(started["emittedAtUtc"], str)
        assert started["traceSessionIdHash"]
        assert started["traceTurnIdHash"]
        assert started["traceMeetingIdHash"]
        assert started["traceOrganizationIdHash"]
        for key in SECRET_LIKE_KEYS:
            assert key not in started

        assert completed["event"] == "request_completed"
        assert completed["stream"] is False
        assert completed["upstreamCreateMs"] >= 0
        assert completed["finishReason"] == "stop"
        assert completed["visibleContentLength"] == len("Hello, I am a helpful assistant.")
        assert completed["contentEmpty"] is False
        assert completed["promptTokens"] == 10
        assert completed["completionTokens"] == 20
        assert completed["totalTokens"] == 30
        assert completed["maxTokensSource"] == "max_tokens"
        assert completed["requestId"] == started["requestId"]
        assert isinstance(completed["emittedAtUtc"], str)
        assert completed["traceSessionIdHash"] == started["traceSessionIdHash"]
        for key in SECRET_LIKE_KEYS:
            assert key not in completed
        trace_blob = json.dumps(traces)
        for raw_value in RAW_TRACE_ID_VALUES:
            assert raw_value not in trace_blob

    def test_streaming_emits_llm_trace_lines(self, capsys):
        _llm_state["client"] = fake_client_instance
        fake_stream = [
            stream_chunk(content="Hello"),
            stream_chunk(content=" world"),
            stream_chunk(content=None, finish_reason="stop"),
        ]
        fake_client_instance.chat.completions.create.return_value = fake_stream

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
                "maxCompletionTokens": 512,
            },
        )
        assert response.status_code == 200

        traces = parse_llm_trace_lines(capsys)
        assert len(traces) == 2
        started, completed = traces

        assert started["event"] == "request_started"
        assert started["stream"] is True
        assert started["maxTokensSource"] == "maxCompletionTokens"
        assert started["forwardedMaxTokens"] == 512
        assert isinstance(started["requestId"], str)
        assert started["requestId"].startswith("chatcmpl-")
        assert isinstance(started["emittedAtUtc"], str)

        assert completed["event"] == "request_completed"
        assert completed["stream"] is True
        assert completed["firstUpstreamChunkMs"] >= 0
        assert completed["firstVisibleContentChunkMs"] >= 0
        assert completed["upstreamChunkCount"] == 3
        assert completed["visibleContentChunkCount"] == 2
        assert completed["totalDurationMs"] >= 0
        assert completed["visibleContentLength"] == len("Hello world")
        assert completed["contentEmpty"] is False
        assert completed["finishReason"] == "stop"
        assert completed["requestId"] == started["requestId"]
        assert isinstance(completed["emittedAtUtc"], str)
        for key in SECRET_LIKE_KEYS:
            assert key not in completed

    def test_upstream_error_emits_request_failed_trace(self, capsys):
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.side_effect = RuntimeError(
            "upstream unavailable"
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 502

        traces = parse_llm_trace_lines(capsys)
        assert len(traces) == 2
        started, failed = traces
        assert started["event"] == "request_started"
        assert failed["event"] == "request_failed"
        assert failed["exceptionClass"] == "RuntimeError"
        assert "upstream unavailable" in failed["exceptionMessage"]
        assert isinstance(started["requestId"], str)
        assert failed["requestId"] == started["requestId"]
        assert isinstance(failed["emittedAtUtc"], str)

    def test_trace_without_headers_omits_hashes_but_keeps_request_id(self, capsys):
        _reset_fake_client()
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

        traces = parse_llm_trace_lines(capsys)
        started, completed = traces
        assert started["requestId"].startswith("chatcmpl-")
        assert completed["requestId"] == started["requestId"]
        assert "traceSessionIdHash" not in started
        assert "traceTurnIdHash" not in started
        assert "traceMeetingIdHash" not in started
        assert "traceOrganizationIdHash" not in started


def _reset_fake_client():
    fake_client_instance.chat.completions.create.side_effect = None
    fake_client_instance.chat.completions.create.return_value = fake_response


class TestThinkingTypeEnv:
    def setup_method(self):
        self._saved_thinking_type = os.environ.pop("LLM_THINKING_TYPE", None)
        _reset_fake_client()

    def teardown_method(self):
        if self._saved_thinking_type is None:
            os.environ.pop("LLM_THINKING_TYPE", None)
        else:
            os.environ["LLM_THINKING_TYPE"] = self._saved_thinking_type

    def test_default_unchanged_when_env_unset(self):
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
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert "extra_body" not in call_kwargs
        assert "reasoning_effort" not in call_kwargs

    def test_default_unchanged_when_env_blank(self):
        os.environ["LLM_THINKING_TYPE"] = "   "
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
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert "extra_body" not in call_kwargs
        assert "reasoning_effort" not in call_kwargs

    def test_disabled_forwards_exact_extra_body(self):
        os.environ["LLM_THINKING_TYPE"] = "disabled"
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
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_enabled_forwards_exact_extra_body(self):
        os.environ["LLM_THINKING_TYPE"] = "enabled"
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
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}

    def test_invalid_value_ignored(self, capsys):
        os.environ["LLM_THINKING_TYPE"] = "off"
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
        call_kwargs = fake_client_instance.chat.completions.create.call_args.kwargs
        assert "extra_body" not in call_kwargs

        traces = parse_llm_trace_lines(capsys)
        started, completed = traces
        assert started["thinkingTypeConfigured"] is False
        assert started["thinkingType"] is None
        assert completed["thinkingTypeConfigured"] is False
        assert completed["thinkingType"] is None
        assert "off" not in json.dumps(traces)


class TestReasoningContentNonLeak:
    def setup_method(self):
        _reset_fake_client()

    def test_reasoning_content_not_leaked_in_streaming_sse_or_trace(self, capsys):
        _llm_state["client"] = fake_client_instance
        fake_stream = [
            stream_chunk(reasoning_content="hidden reasoning"),
            stream_chunk(content="visible answer"),
            stream_chunk(content=None, finish_reason="stop"),
        ]
        fake_client_instance.chat.completions.create.return_value = fake_stream

        with patch("services.llm.main.post_trace_event", new_callable=AsyncMock) as post_trace:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Think"}],
                    "stream": True,
                },
                headers=trace_headers(),
            )

        assert response.status_code == 200
        raw_body = response.content.decode("utf-8")
        assert "hidden reasoning" not in raw_body
        assert "reasoning_content" not in raw_body
        assert "visible answer" in raw_body

        traces = parse_llm_trace_lines(capsys)
        completed = next(entry for entry in traces if entry["event"] == "request_completed")
        assert completed["visibleContentChunkCount"] == 1
        assert completed["visibleContentLength"] == len("visible answer")
        assert "hidden reasoning" not in json.dumps(traces)
        assert "reasoning_content" not in json.dumps(traces)

        post_trace.assert_awaited_once()
        step = post_trace.call_args.kwargs["step"]
        assert step["text"] == "visible answer"
        assert "hidden reasoning" not in json.dumps(step)
        assert "reasoning_content" not in json.dumps(step)


class TestThinkingTypeTraceFields:
    def setup_method(self):
        self._saved_thinking_type = os.environ.pop("LLM_THINKING_TYPE", None)
        _reset_fake_client()

    def teardown_method(self):
        if self._saved_thinking_type is None:
            os.environ.pop("LLM_THINKING_TYPE", None)
        else:
            os.environ["LLM_THINKING_TYPE"] = self._saved_thinking_type

    def test_trace_fields_present_and_safe_when_configured(self, capsys):
        os.environ["LLM_THINKING_TYPE"] = "disabled"
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.return_value = fake_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
            headers=trace_headers(),
        )
        assert response.status_code == 200

        traces = parse_llm_trace_lines(capsys)
        assert len(traces) == 2
        for entry in traces:
            assert "thinkingTypeConfigured" in entry
            assert "thinkingType" in entry
            assert entry["thinkingTypeConfigured"] is True
            assert entry["thinkingType"] == "disabled"
            for key in SECRET_LIKE_KEYS:
                assert key not in entry

    def test_trace_fields_on_request_failed(self, capsys):
        os.environ["LLM_THINKING_TYPE"] = "enabled"
        _llm_state["client"] = fake_client_instance
        fake_client_instance.chat.completions.create.side_effect = RuntimeError("boom")

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 502

        traces = parse_llm_trace_lines(capsys)
        failed = next(entry for entry in traces if entry["event"] == "request_failed")
        assert failed["thinkingTypeConfigured"] is True
        assert failed["thinkingType"] == "enabled"
        for key in SECRET_LIKE_KEYS:
            assert key not in failed
