import json
from unittest.mock import MagicMock, patch

from services.ai_debug import build_trace_body, parse_trace_context, post_trace_event, TraceContext


def test_parse_trace_context_and_post_path():
    ctx = parse_trace_context({
        "X-AI-Trace-Enabled": "true",
        "X-AI-Trace-Session-Id": "session-1",
        "X-AI-Trace-Turn-Id": "turn-1",
        "X-AI-Trace-Sequence-Base": "10",
        "X-AI-Trace-Meeting-Id": "meeting-1",
        "X-AI-Trace-Organization-Id": "org-1",
        "X-AI-Trace-Backend-Url": "http://api:8080/",
        "X-AI-Trace-Agent-Token": "token-secret",
        "X-AI-Trace-Persist-Payloads": "true",
    })
    assert ctx is not None
    assert ctx.persist_payloads is True
    body = build_trace_body(ctx, "stt_completed", {"type": "stt"})
    assert body["sequence"] == 20

    fake_response = MagicMock()
    fake_response.__enter__.return_value.status = 200
    with patch("services.ai_debug.urllib_request.urlopen", return_value=fake_response) as urlopen:
        import asyncio
        assert asyncio.run(post_trace_event(ctx, "stt_completed", {"type": "stt"})) is True

    req = urlopen.call_args.args[0]
    assert req.full_url == "http://api:8080/api/agent/meetings/meeting-1/ai-debug/traces"
    assert req.headers["Authorization"] == "Bearer token-secret"
    assert json.loads(req.data.decode())["eventType"] == "stt_completed"
