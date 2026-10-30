"""The one way triage0 calls an LLM.

Model output is untrusted, so validation lives in the base class: every client,
real or fake, supplies only the transport and can't skip the check.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

__all__ = ["LLMAPIError", "LLMClient", "LLMError", "LLMResponseError"]

T = TypeVar("T", bound=BaseModel)


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
