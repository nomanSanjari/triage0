"""Tests for the real client, against a stand-in server on 127.0.0.1 (no paid API)."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import threading
import urllib.error
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from triage0.llm.client import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    HTTPLLMClient,
    LLMAPIError,
    LLMError,
    LLMResponseError,
)

KEY = "sk-test-secret"


class Answer(BaseModel):
    count: int


def envelope(content: Any) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


GOOD: tuple[int, dict[str, str], bytes] = (200, {}, envelope('{"count": 1}'))


@dataclass
class StandIn:
    """A scripted server: each request takes the next (status, headers, body)."""

    script: list[tuple[int, dict[str, str], bytes]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    url: str = ""
    drop_connections: int = 0
    release: threading.Event = field(default_factory=threading.Event)
    hang: bool = False


@pytest.fixture
def start_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    # A proxy on the developer's machine must not see these requests.
    monkeypatch.setenv("NO_PROXY", "*")
    servers: list[tuple[ThreadingHTTPServer, StandIn]] = []

    def start() -> StandIn:
        state = StandIn()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                state.requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "body": json.loads(body),
                    }
                )
                if state.hang:
                    state.release.wait(5)
                    return
                if state.drop_connections:
                    state.drop_connections -= 1
                    self.close_connection = True
                    return
                status, headers, payload = state.script.pop(0)
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        state.url = f"http://127.0.0.1:{server.server_port}/v1"
        servers.append((server, state))
        return state

    yield start

    for server, state in servers:
        state.release.set()
        server.shutdown()
        server.server_close()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff waits instead of really waiting."""

    recorded: list[float] = []
    monkeypatch.setattr("triage0.llm.client.time.sleep", recorded.append)
    return recorded


def make_client(server: StandIn, **overrides: Any) -> HTTPLLMClient:
    return HTTPLLMClient(server.url, "test-model", KEY, **overrides)


def test_sends_the_request_and_returns_the_validated_answer(start_server: Any) -> None:
    server = start_server()
    server.script = [GOOD]

    answer = make_client(server).complete_structured("rules", "report text", Answer)

    assert answer == Answer(count=1)
    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["auth"] == f"Bearer {KEY}"
    assert request["body"]["model"] == "test-model"
    messages = request["body"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "report text"


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirects_are_refused_so_the_key_stays_put(
    code: int, start_server: Any, sleeps: list[float]
) -> None:
    server, elsewhere = start_server(), start_server()
    server.script = [(code, {"Location": elsewhere.url}, b"")]

    with pytest.raises(LLMAPIError, match=f"HTTP {code}"):
        make_client(server).complete_structured("rules", "data", Answer)

    assert elsewhere.requests == []
    assert len(server.requests) == 1


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_http_errors_are_retried(
    code: int, start_server: Any, sleeps: list[float]
) -> None:
    server = start_server()
    server.script = [(code, {}, b""), (code, {}, b""), GOOD]

    answer = make_client(server).complete_structured("rules", "data", Answer)

    assert answer == Answer(count=1)
    assert sleeps == [1, 2]


def test_retries_are_bounded(start_server: Any, sleeps: list[float]) -> None:
    server = start_server()
    server.script = [(503, {}, b"")] * 3

    with pytest.raises(LLMAPIError, match="HTTP 503"):
        make_client(server).complete_structured("rules", "data", Answer)

    assert len(server.requests) == 3


def test_auth_errors_are_not_retried(start_server: Any, sleeps: list[float]) -> None:
    server = start_server()
    server.script = [(401, {}, b"")]

    with pytest.raises(LLMAPIError, match="HTTP 401"):
        make_client(server).complete_structured("rules", "data", Answer)

    assert len(server.requests) == 1
    assert sleeps == []


def test_dropped_connections_are_retried(start_server: Any, sleeps: list[float]) -> None:
    server = start_server()
    server.drop_connections = 1
    server.script = [GOOD]

    answer = make_client(server).complete_structured("rules", "data", Answer)

    assert answer == Answer(count=1)
    assert len(server.requests) == 2


def test_connection_refused_is_retried(sleeps: list[float]) -> None:
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    client = HTTPLLMClient(f"http://127.0.0.1:{port}/v1", "test-model", KEY)

    with pytest.raises(LLMAPIError, match="ConnectionRefusedError"):
        client.complete_structured("rules", "data", Answer)

    assert sleeps == [1, 2]


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError(socket.gaierror(-3, "try again")),
        http.client.IncompleteRead(b""),
    ],
    ids=["dns", "cut-off-body"],
)
def test_other_transient_failures_are_retried(
    error: Exception, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr("urllib.request.OpenerDirector.open", fail)

    with pytest.raises(LLMAPIError, match=type(getattr(error, "reason", error)).__name__):
        HTTPLLMClient("https://llm.example.test", "m", KEY).complete_structured(
            "rules", "data", Answer
        )

    assert sleeps == [1, 2]


@pytest.mark.parametrize(
    "base_url",
    ["ftp://llm.example.test", "file:///etc/passwd", "llm.example.test/v1"],
    ids=["ftp", "file", "no-scheme"],
)
def test_only_http_base_urls_are_accepted(base_url: str) -> None:
    with pytest.raises(LLMError, match="must start with http"):
        HTTPLLMClient(base_url, "m", KEY)


def test_a_bad_port_fails_at_once(sleeps: list[float]) -> None:
    client = HTTPLLMClient("https://llm.example.test:abc", "m", KEY)

    with pytest.raises(LLMAPIError, match="InvalidURL"):
        client.complete_structured("rules", "data", Answer)

    assert sleeps == []


def test_an_untrusted_certificate_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise urllib.error.URLError(ssl.SSLCertVerificationError("self-signed"))

    monkeypatch.setattr("urllib.request.OpenerDirector.open", fail)

    with pytest.raises(LLMAPIError, match="SSLCertVerificationError"):
        HTTPLLMClient("https://llm.example.test", "m", KEY).complete_structured(
            "rules", "data", Answer
        )

    assert sleeps == []


def test_a_hanging_server_times_out(start_server: Any, sleeps: list[float]) -> None:
    server = start_server()
    server.hang = True

    with pytest.raises(LLMAPIError, match="TimeoutError"):
        make_client(server, timeout=0.1, max_retries=1).complete_structured(
            "rules", "data", Answer
        )

    assert len(server.requests) == 2


def test_bad_model_output_is_not_retried(start_server: Any, sleeps: list[float]) -> None:
    server = start_server()
    server.script = [(200, {}, envelope("not JSON"))]

    with pytest.raises(LLMResponseError):
        make_client(server).complete_structured("rules", "data", Answer)

    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "body",
    [b"<html>", b"{}", b'{"choices": []}', b"[]", b'{"choices": [{}]}'],
    ids=["html", "no-choices", "empty-choices", "wrong-root", "no-message"],
)
def test_an_odd_api_response_raises_a_typed_error(
    body: bytes, start_server: Any, sleeps: list[float]
) -> None:
    server = start_server()
    server.script = [(200, {}, body)]

    with pytest.raises(LLMAPIError, match="unexpected response"):
        make_client(server).complete_structured("rules", "data", Answer)


def test_an_answer_with_no_text_is_a_response_error(
    start_server: Any, sleeps: list[float]
) -> None:
    server = start_server()
    server.script = [(200, {}, envelope(None))]

    with pytest.raises(LLMResponseError, match="no text"):
        make_client(server).complete_structured("rules", "data", Answer)


def test_the_key_never_reaches_errors_or_repr(
    start_server: Any, sleeps: list[float]
) -> None:
    server = start_server()
    server.script = [(401, {}, f'{{"error": "bad key {KEY}"}}'.encode())]
    client = make_client(server)

    with pytest.raises(LLMAPIError) as caught:
        client.complete_structured("rules", "data", Answer)

    assert KEY not in str(caught.value)
    assert KEY not in repr(caught.value)
    assert KEY not in repr(client)


def test_from_env_reads_the_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIAGE0_ENV_FILE", str(tmp_path / "no-such-file"))
    monkeypatch.setenv(ENV_BASE_URL, "https://llm.example.test/v1")
    monkeypatch.setenv(ENV_MODEL, "some-model")
    monkeypatch.setenv(ENV_API_KEY, KEY)

    client = HTTPLLMClient.from_env()

    assert (client.base_url, client.model, client.api_key) == (
        "https://llm.example.test/v1",
        "some-model",
        KEY,
    )


def test_from_env_names_what_is_missing_without_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIAGE0_ENV_FILE", str(tmp_path / "no-such-file"))
    monkeypatch.delenv(ENV_BASE_URL, raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)
    monkeypatch.setenv(ENV_API_KEY, KEY)

    with pytest.raises(LLMError) as caught:
        HTTPLLMClient.from_env()

    assert ENV_BASE_URL in str(caught.value)
    assert ENV_MODEL in str(caught.value)
    assert KEY not in str(caught.value)
