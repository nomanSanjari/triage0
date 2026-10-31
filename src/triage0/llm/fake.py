"""A scripted LLM client, so tests run offline, free and deterministically.

It lives in ``src/`` because the eval harness's offline replay needs it too.
"""

from __future__ import annotations

from triage0.llm.client import LLMClient, LLMError

__all__ = ["FakeLLMClient"]


class FakeLLMClient(LLMClient):
    """Answers each call with the next scripted response.

    A string is returned as the model's raw answer, so malformed output goes
    through the same validation as a real answer. An exception is raised, to
    simulate an API failure.
    """

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def _send(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self._responses:
            raise LLMError("FakeLLMClient ran out of scripted responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
