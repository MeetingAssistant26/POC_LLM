import http.client
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock


MODULE_PATH = Path(__file__).resolve().parents[1] / "wait_for_services.py"
spec = importlib.util.spec_from_file_location("wait_for_services", MODULE_PATH)
wait_for_services = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = wait_for_services
spec.loader.exec_module(wait_for_services)


def test_probe_treats_empty_reply_as_retryable_startup_race(monkeypatch):
    def raise_remote_disconnected(_url, timeout):
        raise http.client.RemoteDisconnected("remote end closed connection without response")

    monkeypatch.setattr(wait_for_services.urllib.request, "urlopen", raise_remote_disconnected)

    ok, detail = wait_for_services._probe("http://localhost:8001/readyz", timeout=1)

    assert ok is False
    assert "startup race: RemoteDisconnected" in detail


def test_wait_for_services_retries_until_all_probes_are_healthy(monkeypatch):
    attempts = {"count": 0}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self, _size):
            return b'{"status":"ok"}'

    def fake_urlopen(_url, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise http.client.RemoteDisconnected("cold startup race")
        return FakeResponse()

    monkeypatch.setattr(wait_for_services.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(wait_for_services.time, "sleep", MagicMock())

    wait_for_services.wait_for_services(
        (wait_for_services.Probe("stt", "http://localhost:8001/readyz"),),
        timeout=5,
        interval=0.01,
        request_timeout=1,
    )

    assert attempts["count"] == 2
