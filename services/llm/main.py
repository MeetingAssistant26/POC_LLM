import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import AliasChoices, BaseModel, Field

from services.ai_debug import duration_ms, parse_trace_context, post_trace_event

load_dotenv()

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _env_value(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return fallback
    return value.strip()


def _llm_base_url() -> str:
    return _env_value("LLM_BASE_URL", _DEFAULT_BASE_URL) or _DEFAULT_BASE_URL


def _llm_api_key() -> str | None:
    # GROQ_API_KEY is kept only as a backward-compatible fallback for existing
    # local environments. New deployments should set LLM_API_KEY.
    return _env_value("LLM_API_KEY", _env_value("GROQ_API_KEY"))


def _llm_model() -> str:
    return _env_value("LLM_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL


# Singleton state for the OpenAI-compatible chat-completions client
_llm_state = {
    "client": None,
    "api_key": _llm_api_key(),
    "base_url": _llm_base_url(),
    "model_name": _llm_model(),
}


def _init_llm_client():
    """Initialize OpenAI-compatible client once and cache it globally."""
    if _llm_state["client"] is None:
        api_key = _llm_api_key()
        base_url = _llm_base_url()
        model_name = _llm_model()
        if not api_key:
            raise RuntimeError("LLM_API_KEY environment variable is not set")
        if not model_name:
            raise RuntimeError("LLM_MODEL environment variable is not set")

        _llm_state["client"] = OpenAI(api_key=api_key, base_url=base_url)
        _llm_state["api_key"] = api_key
        _llm_state["base_url"] = base_url
        _llm_state["model_name"] = model_name
        print(
            "[LLM] OpenAI-compatible client initialized successfully "
            f"for base URL {base_url} and model {model_name}."
        )
    return _llm_state["client"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize OpenAI-compatible client at startup."""
    try:
        _init_llm_client()
    except RuntimeError as e:
        print(f"[LLM] WARNING: {e}")
    yield
    # Optional cleanup on shutdown
    _llm_state["client"] = None


app = FastAPI(
    title="LLM Service",
    description="OpenAI-compatible chat completions proxy.",
    version="1.0.0",
    lifespan=lifespan,
)


def _resolve_model(_model: Optional[str]) -> str:
    """Use the configured upstream model for all proxied requests."""
    return str(_llm_state.get("model_name") or _llm_model())


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("stream_options", "streamOptions"),
    )
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None


def _build_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _build_non_streaming_response(
    upstream_response, model: str, completion_id: str
) -> dict:
    choice = upstream_response.choices[0]
    usage = getattr(upstream_response, "usage", None)
    usage_dict = {}
    if usage:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content,
                },
                "finish_reason": choice.finish_reason,
            }
        ],
        "usage": usage_dict,
    }


def _usage_to_dict(usage: Any) -> dict[str, int] | None:
    if not usage:
        return None

    def token_value(name: str) -> int | None:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        return value if isinstance(value, int) else None

    usage_dict = {
        "prompt_tokens": token_value("prompt_tokens"),
        "completion_tokens": token_value("completion_tokens"),
        "total_tokens": token_value("total_tokens"),
    }
    if all(value is None for value in usage_dict.values()):
        return None
    return {key: value or 0 for key, value in usage_dict.items()}


def _build_stream_chunk(
    completion_id: str,
    model: str,
    delta: dict,
    finish_reason: Optional[str] = None,
    usage: dict[str, int] | None = None,
    choices: list[dict[str, Any]] | None = None,
) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices
        if choices is not None
        else [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    import json

    return f"data: {json.dumps(chunk)}\n\n"


async def _stream_response(
    upstream_stream,
    model: str,
    completion_id: str,
    trace_ctx=None,
    request_payload: dict | None = None,
    start: float | None = None,
) -> AsyncGenerator[str, None]:
    collected_content: list[str] = []
    final_payload: dict = {"id": completion_id, "model": model, "streamed": True}

    # First chunk with role
    yield _build_stream_chunk(
        completion_id, model, {"role": "assistant"}, finish_reason=None
    )

    for chunk in upstream_stream:
        usage = _usage_to_dict(getattr(chunk, "usage", None))
        if usage is not None:
            final_payload["usage"] = usage

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            if usage is not None:
                yield _build_stream_chunk(
                    completion_id, model, {}, usage=usage, choices=[]
                )
            continue

        choice = choices[0]
        delta = choice.delta
        content = getattr(delta, "content", "") or ""
        finish_reason = choice.finish_reason

        if content:
            collected_content.append(content)
            yield _build_stream_chunk(
                completion_id,
                model,
                {"content": content},
                finish_reason=None,
                usage=usage,
            )

        if finish_reason:
            final_payload["finish_reason"] = finish_reason
            yield _build_stream_chunk(
                completion_id, model, {}, finish_reason=finish_reason, usage=usage
            )

    yield "data: [DONE]\n\n"
    final_text = "".join(collected_content)
    usage_payload = final_payload.get("usage", {})
    step = {
        "type": "llm",
        "provider": "OpenAI-compatible",
        "endpoint": "/v1/chat/completions",
        "model": model,
        "durationMs": duration_ms(start or time.perf_counter()),
        "promptTokens": usage_payload.get("prompt_tokens"),
        "completionTokens": usage_payload.get("completion_tokens"),
        "totalTokens": usage_payload.get("total_tokens"),
    }
    if trace_ctx and trace_ctx.persist_payloads:
        step["requestPayload"] = request_payload
        step["responsePayload"] = {**final_payload, "content": final_text}
        step["text"] = final_text
    await post_trace_event(trace_ctx, "llm_completed", step=step)


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Kubernetes-style health probe."""
    healthy = _llm_state["client"] is not None
    if not healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "client_initialized": False},
        )
    return {
        "status": "ok",
        "client_initialized": True,
        "base_url": _llm_state.get("base_url"),
        "model": _llm_state.get("model_name"),
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request, completion_request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Proxies requests to the configured OpenAI-compatible upstream API with
    streaming and non-streaming support.
    """
    trace_ctx = parse_trace_context(request.headers)
    start = time.perf_counter()

    client = _llm_state["client"]
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM client is not initialized. Check LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.",
        )

    model = _resolve_model(completion_request.model)
    completion_id = _build_completion_id()

    # Build kwargs for OpenAI-compatible chat-completions API call.
    kwargs = {
        "model": model,
        "messages": [msg.model_dump() for msg in completion_request.messages],
        "stream": completion_request.stream,
    }
    if completion_request.temperature is not None:
        kwargs["temperature"] = completion_request.temperature
    if completion_request.max_tokens is not None:
        kwargs["max_tokens"] = completion_request.max_tokens
    if completion_request.top_p is not None:
        kwargs["top_p"] = completion_request.top_p
    if completion_request.frequency_penalty is not None:
        kwargs["frequency_penalty"] = completion_request.frequency_penalty
    if completion_request.presence_penalty is not None:
        kwargs["presence_penalty"] = completion_request.presence_penalty
    if completion_request.stop is not None:
        kwargs["stop"] = completion_request.stop

    try:
        upstream_response = client.chat.completions.create(**kwargs)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API error: {str(e)}",
        )

    try:
        request_payload = await request.json()
    except Exception:
        request_payload = completion_request.model_dump(exclude_none=True)

    if completion_request.stream:
        return StreamingResponse(
            _stream_response(upstream_response, model, completion_id, trace_ctx, request_payload, start),
            media_type="text/event-stream",
        )

    response_body = _build_non_streaming_response(upstream_response, model, completion_id)
    usage_payload = response_body.get("usage", {})
    final_text = response_body["choices"][0]["message"]["content"]
    step = {
        "type": "llm",
        "provider": "OpenAI-compatible",
        "endpoint": str(request.url),
        "model": model,
        "durationMs": duration_ms(start),
        "promptTokens": usage_payload.get("prompt_tokens"),
        "completionTokens": usage_payload.get("completion_tokens"),
        "totalTokens": usage_payload.get("total_tokens"),
    }
    if trace_ctx and trace_ctx.persist_payloads:
        step["requestPayload"] = request_payload
        step["responsePayload"] = response_body
        step["text"] = final_text
    await post_trace_event(trace_ctx, "llm_completed", step=step)
    return response_body
