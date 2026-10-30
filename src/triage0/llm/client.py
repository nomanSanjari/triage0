"""The one way triage0 calls an LLM.

Model output is untrusted, so validation lives in the base class: every client,
real or fake, supplies only the transport and can't skip the check.

``HTTPLLMClient`` sends plain HTTP in the OpenAI-compatible format, which many
providers accept, so choosing a provider is configuration, not a code change.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import IO, Self, TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from triage0.config import load_env

__all__ = [
    "HTTPLLMClient",
    "LLMAPIError",
    "LLMClient",
    "LLMError",
    "LLMResponseError",
]

T = TypeVar("T", bound=BaseModel)

ENV_API_KEY = "TRIAGE0_LLM_API_KEY"
ENV_MODEL = "TRIAGE0_LLM_MODEL"
ENV_BASE_URL = "TRIAGE0_LLM_BASE_URL"

RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})
# An allowlist, so a failure that can't fix itself (an untrusted certificate, a
# bad URL) fails at once instead of being retried.
TRANSIENT_ERRORS = (
    TimeoutError,
    ConnectionError,
    socket.gaierror,
    http.client.IncompleteRead,
)
MAX_BACKOFF_SECONDS = 8


class LLMError(Exception):
    """Base class, so callers can handle every LLM failure in one place."""


class LLMResponseError(LLMError):
    """The model answered, but not with what was asked for. Never retried."""


class LLMAPIError(LLMError):
    """The call itself failed: HTTP error, network error or timeout."""


class LLMClient(ABC):
    def complete_structured(self, system: str, user: str, schema: type[T]) -> T:
        """Ask the model for JSON and return it validated against ``schema``.

        ``system`` is our instructions and ``user`` is the data to work on.
        They stay separate so report text is never mixed into the instructions.
        Bad output raises ``LLMResponseError``, never a default value: an empty
        result must always mean the model found nothing, not that a call failed.
        """

        instructions = (
            f"{system}\n\nReturn only JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        text = self._send(instructions, user)
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMResponseError(
                f"model output did not match {schema.__name__}: "
                f"{exc.error_count()} error(s)"
            ) from None

    @abstractmethod
    def _send(self, system: str, user: str) -> str:
        """Return the model's raw answer, or raise ``LLMAPIError``."""


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Treat every redirect as an error.

    Following one could send the API key to another server, and a model API
    has no reason to redirect, so a wrong base URL should simply fail loudly.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> None:
        return None


def _answer_text(raw: bytes) -> str:
    """Pull the model's answer out of the provider's response envelope."""

    try:
        content = json.loads(raw)["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMAPIError("model API returned an unexpected response") from None
    if not isinstance(content, str):
        raise LLMResponseError("model returned no text")
    return content


@dataclass(frozen=True)
class HTTPLLMClient(LLMClient):
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout: float = 120.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        # urllib also speaks ftp:// and file://, which have no place here.
        if urlsplit(self.base_url).scheme not in ("http", "https"):
            raise LLMError("the LLM base URL must start with http:// or https://")

    @classmethod
    def from_env(cls) -> Self:
        """Build a client from the environment or ``.env``.

        There are deliberately no defaults: a guessed provider could receive
        the key by mistake.
        """

        load_env()
        names = (ENV_BASE_URL, ENV_MODEL, ENV_API_KEY)
        missing = [name for name in names if not os.getenv(name)]
        if missing:
            raise LLMError(f"missing configuration: {', '.join(missing)}")
        return cls(os.environ[ENV_BASE_URL], os.environ[ENV_MODEL], os.environ[ENV_API_KEY])

    def _send(self, system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(_RefuseRedirects)

        for attempt in range(self.max_retries + 1):
            is_last = attempt == self.max_retries
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    return _answer_text(response.read())
            except urllib.error.HTTPError as exc:
                exc.close()
                if exc.code not in RETRYABLE_HTTP_CODES or is_last:
                    # Only the status: provider error text can echo the key.
                    raise LLMAPIError(f"model API returned HTTP {exc.code}") from None
            except (OSError, http.client.HTTPException) as exc:
                # urllib wraps connection failures, hiding the real cause.
                cause = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                if is_last or not isinstance(cause, TRANSIENT_ERRORS):
                    raise LLMAPIError(
                        f"model API request failed: {type(cause).__name__}"
                    ) from None
            time.sleep(min(2**attempt, MAX_BACKOFF_SECONDS))
        raise AssertionError("unreachable: the last attempt returns or raises")
