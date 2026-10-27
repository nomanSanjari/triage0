"""Redirect handling for the baseline HTTP client.

These tests use a fake opener parent, so no network access is required. They
cover the original crash (a rebuilt request lost ``req.timeout``) and the
credential-safety rules: same-origin redirects follow, cross-origin and
HTTPS->HTTP redirects are refused.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from email.message import Message

import pytest

from triage0.eval.baseline import baseline as baseline_module
from triage0.eval.baseline.baseline import _SameOriginRedirectHandler

ORIGIN = "https://api.example.test/v1/chat/completions"


class _FakeParent:
    def __init__(self) -> None:
        self.calls: list[tuple[urllib.request.Request, float | None]] = []

    def open(
        self, request: urllib.request.Request, timeout: float | None = None
    ) -> str:
        self.calls.append((request, timeout))
        return "response"


def _handler(url: str = ORIGIN) -> tuple[_SameOriginRedirectHandler, _FakeParent]:
    handler = _SameOriginRedirectHandler(url)
    parent = _FakeParent()
    handler.parent = parent  # type: ignore[assignment]
    return handler, parent


def _request(
    url: str = ORIGIN,
    *,
    method: str = "POST",
    data: bytes | None = b'{"model":"test"}',
) -> urllib.request.Request:
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": "Bearer secret"},
        method=method,
    )
    # OpenerDirector.open() sets this before the redirect handler runs.
    request.timeout = 5.0
    return request


def _headers(location: str | None) -> Message:
    headers = Message()
    if location is not None:
        headers["Location"] = location
    return headers


def test_same_origin_302_resolves_relative_location_and_keeps_credentials() -> None:
    handler, parent = _handler()
    result = handler.http_error_302(
        _request(), io.BytesIO(b""), 302, "Found", _headers("/v2/chat")
    )

    assert result == "response"
    redirected, timeout = parent.calls[0]
    assert redirected.full_url == "https://api.example.test/v2/chat"
    assert redirected.get_header("Authorization") == "Bearer secret"
    # The original crash: super().http_error_302() reads req.timeout from the
    # request it is given, so the rebuilt request must carry it.
    assert timeout == 5.0


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_cross_origin_redirect_is_refused(code: int) -> None:
    handler, parent = _handler()
    method = getattr(handler, f"http_error_{code}")

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        method(
            _request(),
            io.BytesIO(b""),
            code,
            "Redirect",
            _headers("https://evil.example/steal"),
        )

    assert excinfo.value.code == 403
    assert "Cross-origin" in excinfo.value.reason
    assert parent.calls == []


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_https_to_http_downgrade_is_refused(code: int) -> None:
    handler, parent = _handler()
    method = getattr(handler, f"http_error_{code}")

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        method(
            _request(),
            io.BytesIO(b""),
            code,
            "Redirect",
            _headers("http://api.example.test/v1/chat/completions"),
        )

    assert excinfo.value.code == 403
    assert "downgrade" in excinfo.value.reason
    assert parent.calls == []


def test_same_origin_307_preserves_method_and_body() -> None:
    handler, parent = _handler()
    result = handler.http_error_307(
        _request(),
        io.BytesIO(b""),
        307,
        "Temporary Redirect",
        _headers("https://api.example.test/v2/chat"),
    )

    assert result == "response"
    redirected, timeout = parent.calls[0]
    assert redirected.full_url == "https://api.example.test/v2/chat"
    assert redirected.get_method() == "POST"
    assert redirected.data == b'{"model":"test"}'
    assert redirected.get_header("Authorization") == "Bearer secret"
    assert timeout == 5.0


def test_redirect_without_location_is_ignored() -> None:
    handler, parent = _handler()
    result = handler.http_error_302(
        _request(), io.BytesIO(b""), 302, "Found", _headers(None)
    )

    assert result is None
    assert parent.calls == []


def test_same_origin_redirect_requires_a_known_origin() -> None:
    handler, parent = _handler(url="")
    # A handler without an origin must fail closed rather than follow blindly.
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        handler.http_error_302(
            _request(),
            io.BytesIO(b""),
            302,
            "Found",
            _headers("https://api.example.test/v2/chat"),
        )
    assert excinfo.value.code == 403
    assert parent.calls == []


def test_call_model_reports_redirect_refusal_as_baseline_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RefusingOpener:
        def open(
            self, request: urllib.request.Request, timeout: float | None = None
        ) -> object:
            raise urllib.error.HTTPError(
                request.full_url,
                403,
                "Cross-origin redirect refused",
                Message(),
                io.BytesIO(b"refused"),
            )

    monkeypatch.setattr(
        baseline_module.urllib.request,
        "build_opener",
        lambda *args, **kwargs: RefusingOpener(),
    )

    with pytest.raises(baseline_module.BaselineError, match="HTTP 403"):
        baseline_module.call_model("prompt", api_key="test", max_retries=0)
