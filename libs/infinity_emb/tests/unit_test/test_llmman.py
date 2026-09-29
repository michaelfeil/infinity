"""The `llmman serve` client: the daemon protocol behind oci:// model paths.

Exercised against a real HTTP server on a loopback port rather than mocks, so
the NDJSON streaming contract is genuinely tested.
"""

import http.server
import json
import socketserver
import subprocess
import sys
import threading
from unittest import mock

import pytest

from infinity_emb import llmman


def _ndjson(*objs):
    return "".join(json.dumps(o) + "\n" for o in objs)


class _FakeDaemon:
    """A minimal stand-in for `llmman serve`, on a real loopback port."""

    def __init__(self):
        self.version = {"version": "0.1.0", "pid": 1}
        self.pull_body = _ndjson({"status": "success"})
        self.pull_status = 200
        self.last_request = None
        self.stall = None  # "headers": never answer, "stream": answer then go quiet
        self.release = threading.Event()
        daemon = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body, ctype):
                raw = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if daemon.stall == "headers":
                    return daemon.release.wait(10)
                self._send(200, json.dumps(daemon.version), "application/json")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                daemon.last_request = json.loads(self.rfile.read(length))
                if daemon.stall == "headers":
                    return daemon.release.wait(10)
                if daemon.stall == "stream":
                    self.send_response(200)
                    self.send_header("Content-Length", "1000")
                    self.end_headers()
                    self.wfile.write(_ndjson({"status": "pulling manifest"}).encode())
                    return daemon.release.wait(10)
                self._send(daemon.pull_status, daemon.pull_body, "application/x-ndjson")

        self._server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self.release.set()
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def daemon():
    d = _FakeDaemon()
    yield d
    d.close()


def test_accepts_a_llmman_daemon(daemon):
    llmman.check_daemon(daemon.url)


def test_rejects_a_non_llmman_server(daemon):
    daemon.version = {"hello": "world"}
    with pytest.raises(RuntimeError, match="not an llmman daemon"):
        llmman.check_daemon(daemon.url)


def test_reports_nothing_listening_actionably():
    with pytest.raises(RuntimeError, match="llmman serve"):
        llmman.check_daemon("http://127.0.0.1:1")


def test_pull_succeeds_and_forwards_progress(daemon):
    daemon.pull_body = _ndjson(
        {"status": "pulling manifest"},
        {"status": "pulling blobs", "completed": 50, "total": 100},
        {"status": "success"},
    )
    seen = []
    llmman.pull(daemon.url, "ghcr.io/org/model:tag", lambda *a: seen.append(a))

    assert daemon.last_request == {"model": "ghcr.io/org/model:tag"}
    assert seen == [("pulling manifest", 0, 0), ("pulling blobs", 50, 100)]


def test_reports_an_in_band_error_at_http_200(daemon):
    # The daemon streams errors in-band, so a 200 does not mean success.
    daemon.pull_body = _ndjson({"status": "pulling"}, {"error": "unauthorized"})
    with pytest.raises(RuntimeError, match="unauthorized"):
        llmman.pull(daemon.url, "ref")


def test_rejects_a_stream_that_ends_without_success(daemon):
    daemon.pull_body = _ndjson({"status": "pulling blobs"})
    with pytest.raises(RuntimeError, match="without reporting success"):
        llmman.pull(daemon.url, "ref")


def test_reports_a_non_ok_status(daemon):
    daemon.pull_status = 400
    daemon.pull_body = '{"error":"bad request"}'
    with pytest.raises(RuntimeError):
        llmman.pull(daemon.url, "ref")


def test_tolerates_a_non_json_diagnostic_line(daemon):
    daemon.pull_body = "not json\n" + _ndjson({"status": "success"})
    llmman.pull(daemon.url, "ref")


@pytest.fixture
def fast_timeouts(monkeypatch):
    monkeypatch.setattr(llmman, "PROBE_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(llmman, "PULL_STALL_TIMEOUT_SECONDS", 0.3)


def test_gives_up_on_a_stalled_probe(daemon, fast_timeouts):
    daemon.stall = "headers"
    with pytest.raises(RuntimeError, match="llmman serve"):
        llmman.check_daemon(daemon.url)


@pytest.mark.parametrize("stall", ["headers", "stream"])
def test_pull_gives_up_on_a_stalled_daemon(daemon, fast_timeouts, stall):
    daemon.stall = stall
    with pytest.raises(RuntimeError, match="timed out"):
        llmman.pull(daemon.url, "ref")


def test_resolve_bounds_the_subprocess():
    hang = subprocess.TimeoutExpired("llmman", 1)
    with mock.patch.object(llmman.subprocess, "run", side_effect=hang) as run:
        with pytest.raises(RuntimeError, match="timed out"):
            llmman.resolve("ref", binary="llmman")
    assert run.call_args.kwargs["timeout"] == llmman.RESOLVE_TIMEOUT_SECONDS


def test_a_missing_binary_fails_before_the_pull(daemon, monkeypatch, tmp_path):
    monkeypatch.setenv(llmman.HOST_ENV, daemon.url)
    monkeypatch.setenv(llmman.BIN_ENV, str(tmp_path / "no-such-llmman"))
    with pytest.raises(RuntimeError, match="not found"):
        llmman.pull_and_resolve("ref")
    assert daemon.last_request is None


def test_pull_and_resolve_pulls_then_resolves(daemon, monkeypatch, tmp_path):
    monkeypatch.setenv(llmman.HOST_ENV, daemon.url)
    monkeypatch.setenv(llmman.BIN_ENV, sys.executable)
    done = subprocess.CompletedProcess([], 0, stdout=json.dumps({"path": str(tmp_path)}))
    with mock.patch.object(llmman.subprocess, "run", return_value=done) as run:
        assert llmman.pull_and_resolve("ref") == str(tmp_path)
    assert daemon.last_request == {"model": "ref"}
    assert run.call_args.args[0] == [sys.executable, "resolve", "--no-pull", "ref"]
