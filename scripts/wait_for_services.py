#!/usr/bin/env python3
"""Wait for AI microservice health/readiness endpoints after Docker compose startup.

This is intentionally stdlib-only so it can run from a fresh checkout without
installing project dependencies. It treats connection resets/empty replies as a
normal startup race after `docker compose up -d --build` and keeps polling until
all endpoints return HTTP 200 or the deadline expires.
"""

from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    name: str
    url: str


DEFAULT_PROBES = (
    Probe("stt", "http://localhost:8001/readyz"),
    Probe("llm", "http://localhost:8002/healthz"),
    Probe("tts", "http://localhost:8003/healthz"),
)


def _probe(url: str, timeout: float) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = response.status
            body = response.read(500)
        if status == 200:
            detail = ""
            if body:
                try:
                    parsed = json.loads(body.decode("utf-8"))
                    detail = f" {parsed}"
                except Exception:
                    detail = f" {body[:120]!r}"
            return True, f"HTTP 200{detail}"
        return False, f"HTTP {status}"
    except urllib.error.HTTPError as exc:
        body = exc.read(200).decode("utf-8", errors="replace")
        return False, f"HTTP {exc.code} {body}".strip()
    except (
        http.client.RemoteDisconnected,
        ConnectionResetError,
        ConnectionRefusedError,
        TimeoutError,
        urllib.error.URLError,
        OSError,
    ) as exc:
        return False, f"startup race: {type(exc).__name__}"


def wait_for_services(probes: tuple[Probe, ...], timeout: float, interval: float, request_timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_results: dict[str, str] = {}
    while time.monotonic() < deadline:
        all_ok = True
        for probe in probes:
            ok, detail = _probe(probe.url, request_timeout)
            last_results[probe.name] = detail
            if not ok:
                all_ok = False
        status_line = "; ".join(f"{name}={detail}" for name, detail in last_results.items())
        print(f"[wait] {status_line}", flush=True)
        if all_ok:
            print("[wait] all AI services are ready", flush=True)
            return
        time.sleep(interval)
    raise TimeoutError(
        f"AI services did not become ready within {timeout:g}s: "
        + "; ".join(f"{name}={detail}" for name, detail in last_results.items())
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for Docker AI services to be ready")
    parser.add_argument("--timeout", type=float, default=420.0, help="overall timeout in seconds")
    parser.add_argument("--interval", type=float, default=5.0, help="poll interval in seconds")
    parser.add_argument("--request-timeout", type=float, default=5.0, help="per-request timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        wait_for_services(DEFAULT_PROBES, args.timeout, args.interval, args.request_timeout)
        return 0
    except Exception as exc:
        print(f"[wait] failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
