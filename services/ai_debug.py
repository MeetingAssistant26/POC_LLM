from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceContext:
    enabled: bool
    session_id: str
    turn_id: str
    sequence_base: int
    meeting_id: str
    organization_id: str
    backend_url: str
    agent_token: str
    persist_payloads: bool


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _header(headers: Mapping[str, str], name: str) -> str:
    value = headers.get(name) or headers.get(name.lower()) or ""
    return value.strip()


def parse_trace_context(headers: Mapping[str, str]) -> Optional[TraceContext]:
    """Read X-AI-Trace-* headers. Returns None unless all required fields are present."""
    if not _truthy(_header(headers, "X-AI-Trace-Enabled")):
        return None

    sequence_base_raw = _header(headers, "X-AI-Trace-Sequence-Base")
    try:
        sequence_base = int(sequence_base_raw) if sequence_base_raw else 0
    except ValueError:
        sequence_base = 0

    ctx = TraceContext(
        enabled=True,
        session_id=_header(headers, "X-AI-Trace-Session-Id"),
        turn_id=_header(headers, "X-AI-Trace-Turn-Id"),
        sequence_base=sequence_base,
        meeting_id=_header(headers, "X-AI-Trace-Meeting-Id"),
        organization_id=_header(headers, "X-AI-Trace-Organization-Id"),
        backend_url=_header(headers, "X-AI-Trace-Backend-Url").rstrip("/"),
        agent_token=_header(headers, "X-AI-Trace-Agent-Token"),
        persist_payloads=_truthy(_header(headers, "X-AI-Trace-Persist-Payloads")),
    )
    if not (ctx.session_id and ctx.turn_id and ctx.meeting_id and ctx.backend_url and ctx.agent_token):
        return None
    return ctx


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def duration_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _sequence(ctx: TraceContext, event_type: str) -> int:
    offsets = {
        "stt_completed": 10,
        "llm_completed": 20,
        "tts_completed": 30,
        "error": 90,
    }
    return ctx.sequence_base + offsets.get(event_type, 0)


def build_trace_body(
    ctx: TraceContext,
    event_type: str,
    step: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    participant_identity: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "sessionId": ctx.session_id,
        "turnId": ctx.turn_id,
        "sequence": _sequence(ctx, event_type),
        "eventType": event_type,
        "occurredAtUtc": utc_now_iso(),
    }
    if participant_identity:
        body["participantIdentity"] = participant_identity
    if state:
        body["state"] = state
    if step is not None:
        body["step"] = step
    if error is not None:
        body["error"] = error
    return body


def _post_trace_sync(ctx: TraceContext, body: dict[str, Any]) -> bool:
    url = f"{ctx.backend_url}/api/agent/meetings/{quote(ctx.meeting_id, safe='')}/ai-debug/traces"
    payload = json.dumps(body).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {ctx.agent_token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib_request.urlopen(req, timeout=2) as resp:
            return 200 <= getattr(resp, "status", 500) < 300
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        logger.warning("AI debug trace POST failed: %s", exc)
        return False


async def post_trace_event(
    ctx: TraceContext | None,
    event_type: str,
    step: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    participant_identity: str | None = None,
    state: str | None = None,
) -> bool:
    if ctx is None:
        return False
    body = build_trace_body(ctx, event_type, step, error, participant_identity, state)
    try:
        return await asyncio.to_thread(_post_trace_sync, ctx, body)
    except Exception as exc:  # defensive: tracing must never fail service requests
        logger.warning("AI debug trace POST failed: %s", exc)
        return False
