import hashlib
import json
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

from services.ai_debug import duration_ms, parse_trace_context, post_trace_event, utc_now_iso

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


_ALLOWED_THINKING_TYPES = frozenset({"enabled", "disabled"})


def _llm_thinking_type() -> str | None:
    value = _env_value("LLM_THINKING_TYPE")
    if value is None:
        return None
    normalized = value.lower()
    if normalized not in _ALLOWED_THINKING_TYPES:
        return None
    return normalized


def _thinking_type_metadata() -> dict[str, bool | str | None]:
    thinking_type = _llm_thinking_type()
    return {
        "thinkingTypeConfigured": thinking_type is not None,
        "thinkingType": thinking_type,
    }


def _thinking_type_extra_body() -> dict[str, Any] | None:
    thinking_type = _llm_thinking_type()
    if thinking_type is None:
        return None
    return {"thinking": {"type": thinking_type}}


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
    max_completion_tokens: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("max_completion_tokens", "maxCompletionTokens"),
    )
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None


def _build_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


_LLM_TRACE_PREFIX = "[LLM_TRACE] "


def _emit_llm_trace(payload: dict[str, Any]) -> None:
    print(f"{_LLM_TRACE_PREFIX}{json.dumps(payload, separators=(',', ':'))}")


def _trace_id_hash(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _trace_context_hashes(trace_ctx: Any) -> dict[str, str | None]:
    if trace_ctx is None:
        return {}
    return {
        "traceSessionIdHash": _trace_id_hash(trace_ctx.session_id),
        "traceTurnIdHash": _trace_id_hash(trace_ctx.turn_id),
        "traceMeetingIdHash": _trace_id_hash(trace_ctx.meeting_id),
        "traceOrganizationIdHash": _trace_id_hash(trace_ctx.organization_id),
    }


def _llm_trace_common_fields(*, completion_id: str, trace_ctx: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "requestId": completion_id,
        "emittedAtUtc": utc_now_iso(),
    }
    fields.update(_trace_context_hashes(trace_ctx))
    return fields


def _trace_context_bools(trace_ctx: Any) -> dict[str, bool]:
    if trace_ctx is None:
        return {
            "traceEnabled": False,
            "traceSessionIdPresent": False,
            "traceTurnIdPresent": False,
            "traceMeetingIdPresent": False,
            "traceOrganizationIdPresent": False,
            "traceBackendUrlPresent": False,
            "traceAgentTokenPresent": False,
            "tracePersistPayloads": False,
        }
    return {
        "traceEnabled": True,
        "traceSessionIdPresent": bool(trace_ctx.session_id),
        "traceTurnIdPresent": bool(trace_ctx.turn_id),
        "traceMeetingIdPresent": bool(trace_ctx.meeting_id),
        "traceOrganizationIdPresent": bool(trace_ctx.organization_id),
        "traceBackendUrlPresent": bool(trace_ctx.backend_url),
        "traceAgentTokenPresent": bool(trace_ctx.agent_token),
        "tracePersistPayloads": bool(trace_ctx.persist_payloads),
    }


def _resolve_max_tokens_metadata(
    request_payload: dict[str, Any] | None,
    completion_request: ChatCompletionRequest,
) -> dict[str, Any]:
    max_tokens_present = completion_request.max_tokens is not None
    max_completion_tokens_present = completion_request.max_completion_tokens is not None

    source = "unset"
    forwarded: int | None = None
    if isinstance(request_payload, dict):
        if request_payload.get("max_tokens") is not None:
            source = "max_tokens"
            forwarded = completion_request.max_tokens
        elif request_payload.get("max_completion_tokens") is not None:
            source = "max_completion_tokens"
            forwarded = completion_request.max_completion_tokens
        elif request_payload.get("maxCompletionTokens") is not None:
            source = "maxCompletionTokens"
            forwarded = completion_request.max_completion_tokens
    elif max_tokens_present:
        source = "max_tokens"
        forwarded = completion_request.max_tokens
    elif max_completion_tokens_present:
        source = "max_completion_tokens"
        forwarded = completion_request.max_completion_tokens

    return {
        "maxTokensPresent": max_tokens_present,
        "maxCompletionTokensPresent": max_completion_tokens_present,
        "maxTokensSource": source,
        "forwardedMaxTokens": forwarded,
    }


def _extract_reasoning_tokens(usage: Any) -> int | None:
    if not usage:
        return None

    def read_reasoning(value: Any) -> int | None:
        if isinstance(value, dict):
            reasoning = value.get("reasoning_tokens")
            return reasoning if isinstance(reasoning, int) else None
        reasoning = getattr(value, "reasoning_tokens", None)
        return reasoning if isinstance(reasoning, int) else None

    direct = read_reasoning(usage)
    if direct is not None:
        return direct

    for details_key in ("completion_tokens_details", "output_tokens_details"):
        if isinstance(usage, dict):
            details = usage.get(details_key)
        else:
            details = getattr(usage, details_key, None)
        reasoning = read_reasoning(details)
        if reasoning is not None:
            return reasoning
    return None


def _usage_trace_fields(usage: Any) -> dict[str, int | None]:
    usage_dict = _usage_to_dict(usage)
    if not usage_dict:
        return {
            "promptTokens": None,
            "completionTokens": None,
            "totalTokens": None,
            "reasoningTokens": _extract_reasoning_tokens(usage),
        }
    return {
        "promptTokens": usage_dict.get("prompt_tokens"),
        "completionTokens": usage_dict.get("completion_tokens"),
        "totalTokens": usage_dict.get("total_tokens"),
        "reasoningTokens": _extract_reasoning_tokens(usage),
    }


async def _read_request_payload(
    request: Request, completion_request: ChatCompletionRequest
) -> dict[str, Any]:
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return completion_request.model_dump(exclude_none=True)


def _build_request_started_trace(
    *,
    model: str,
    completion_id: str,
    completion_request: ChatCompletionRequest,
    request_payload: dict[str, Any],
    trace_ctx: Any,
) -> dict[str, Any]:
    return {
        "event": "request_started",
        "model": model,
        "stream": completion_request.stream,
        "messageCount": len(completion_request.messages),
        **_resolve_max_tokens_metadata(request_payload, completion_request),
        **_thinking_type_metadata(),
        **_trace_context_bools(trace_ctx),
        **_llm_trace_common_fields(completion_id=completion_id, trace_ctx=trace_ctx),
    }


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
    max_tokens_metadata: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    stream_start = start or time.perf_counter()
    collected_content: list[str] = []
    final_payload: dict = {"id": completion_id, "model": model, "streamed": True}
    first_upstream_chunk_ms: int | None = None
    first_visible_content_chunk_ms: int | None = None
    upstream_chunk_count = 0
    visible_content_chunk_count = 0
    final_usage: Any = None

    # First chunk with role
    yield _build_stream_chunk(
        completion_id, model, {"role": "assistant"}, finish_reason=None
    )

    for chunk in upstream_stream:
        upstream_chunk_count += 1
        if first_upstream_chunk_ms is None:
            first_upstream_chunk_ms = duration_ms(stream_start)

        usage = _usage_to_dict(getattr(chunk, "usage", None))
        if usage is not None:
            final_payload["usage"] = usage
            final_usage = getattr(chunk, "usage", None)

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
        # reasoning_content is intentionally ignored and never forwarded to SSE.
        finish_reason = choice.finish_reason

        if content:
            visible_content_chunk_count += 1
            if first_visible_content_chunk_ms is None:
                first_visible_content_chunk_ms = duration_ms(stream_start)
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
    usage_trace = _usage_trace_fields(final_usage)
    _emit_llm_trace(
        {
            "event": "request_completed",
            "stream": True,
            "firstUpstreamChunkMs": first_upstream_chunk_ms,
            "firstVisibleContentChunkMs": first_visible_content_chunk_ms,
            "upstreamChunkCount": upstream_chunk_count,
            "visibleContentChunkCount": visible_content_chunk_count,
            "totalDurationMs": duration_ms(stream_start),
            "finishReason": final_payload.get("finish_reason"),
            "visibleContentLength": len(final_text),
            "contentEmpty": len(final_text) == 0,
            **usage_trace,
            **_thinking_type_metadata(),
            **(max_tokens_metadata or {}),
            **_llm_trace_common_fields(completion_id=completion_id, trace_ctx=trace_ctx),
        }
    )
    step = {
        "type": "llm",
        "provider": "OpenAI-compatible",
        "endpoint": "/v1/chat/completions",
        "model": model,
        "durationMs": duration_ms(stream_start),
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
    request_payload = await _read_request_payload(request, completion_request)
    max_tokens_metadata = _resolve_max_tokens_metadata(request_payload, completion_request)
    _emit_llm_trace(
        _build_request_started_trace(
            model=model,
            completion_id=completion_id,
            completion_request=completion_request,
            request_payload=request_payload,
            trace_ctx=trace_ctx,
        )
    )

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
    elif completion_request.max_completion_tokens is not None:
        kwargs["max_tokens"] = completion_request.max_completion_tokens
    if completion_request.top_p is not None:
        kwargs["top_p"] = completion_request.top_p
    if completion_request.frequency_penalty is not None:
        kwargs["frequency_penalty"] = completion_request.frequency_penalty
    if completion_request.presence_penalty is not None:
        kwargs["presence_penalty"] = completion_request.presence_penalty
    if completion_request.stop is not None:
        kwargs["stop"] = completion_request.stop
    extra_body = _thinking_type_extra_body()
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    try:
        upstream_response = client.chat.completions.create(**kwargs)
    except Exception as e:
        _emit_llm_trace(
            {
                "event": "request_failed",
                "stream": completion_request.stream,
                "exceptionClass": type(e).__name__,
                "exceptionMessage": str(e)[:500],
                **_thinking_type_metadata(),
                **_llm_trace_common_fields(completion_id=completion_id, trace_ctx=trace_ctx),
            }
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API error: {str(e)}",
        )

    if completion_request.stream:
        return StreamingResponse(
            _stream_response(
                upstream_response,
                model,
                completion_id,
                trace_ctx,
                request_payload,
                start,
                max_tokens_metadata,
            ),
            media_type="text/event-stream",
        )

    response_body = _build_non_streaming_response(upstream_response, model, completion_id)
    usage_payload = response_body.get("usage", {})
    final_text = response_body["choices"][0]["message"]["content"] or ""
    finish_reason = response_body["choices"][0].get("finish_reason")
    usage_trace = _usage_trace_fields(getattr(upstream_response, "usage", None))
    _emit_llm_trace(
        {
            "event": "request_completed",
            "stream": False,
            "upstreamCreateMs": duration_ms(start),
            "finishReason": finish_reason,
            "visibleContentLength": len(final_text),
            "contentEmpty": len(final_text) == 0,
            **usage_trace,
            **_thinking_type_metadata(),
            **max_tokens_metadata,
            **_llm_trace_common_fields(completion_id=completion_id, trace_ctx=trace_ctx),
        }
    )
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
