"""Report intake: normalize external formats into ``NormalizedReport``.

The registry is the single place that maps a :class:`ReportSource` to an
adapter, so callers never import concrete adapter modules.  ``email`` and
``markdown`` share one adapter because a mailing-list report is Markdown inside
an RFC 5322 message.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from triage0.core.interfaces import IntakeAdapter
from triage0.core.models import ReportSource
from triage0.intake._common import (
    IntakeError,
    decode_json,
    decode_text,
)
from triage0.intake.github_adapter import GitHubAdvisoryAdapter
from triage0.intake.hackerone_adapter import HackerOneAdapter
from triage0.intake.markdown_adapter import MarkdownAdapter

__all__ = [
    "ADAPTERS",
    "GitHubAdvisoryAdapter",
    "HackerOneAdapter",
    "IntakeError",
    "MarkdownAdapter",
    "adapter_for",
    "detect_source",
]

_MARKDOWN = MarkdownAdapter()

ADAPTERS: dict[ReportSource, IntakeAdapter] = {
    ReportSource.MARKDOWN: _MARKDOWN,
    ReportSource.EMAIL: _MARKDOWN,
    ReportSource.HACKERONE: HackerOneAdapter(),
    ReportSource.GITHUB_ADVISORY: GitHubAdvisoryAdapter(),
}

# Marker keys are searched recursively, because GitHub wraps advisories in
# ``data``/``repository``/``securityAdvisories`` envelopes.  They are chosen to
# be unambiguous between the two JSON formats.
_GITHUB_MARKERS = frozenset(
    {
        "advisory_id",
        "ghsaId",
        "ghsa_id",
        "repositoryAdvisory",
        "securityAdvisories",
        "securityAdvisory",
    }
)
_HACKERONE_STRONG_MARKERS = frozenset(
    {
        "vulnerability_information",
        "database_id",
        "structured_scope",
        "bounty_amount",
    }
)
_HACKERONE_WEAK_MARKERS = frozenset(
    {
        "team",
        "weakness",
    }
)

_KEY_SEARCH_DEPTH = 5


def adapter_for(source: ReportSource) -> IntakeAdapter:
    """Return the adapter registered for ``source``."""

    adapter = ADAPTERS.get(source)
    if adapter is None:
        raise IntakeError(
            f"no intake adapter is registered for {source.value!r}"
        )
    return adapter


def _payload_object(
    payload: str | bytes | Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a JSON object if the payload is JSON, else ``None``."""

    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, (str, bytes)):
        text = decode_text(payload)
        if not text.lstrip().startswith("{"):
            if text.lstrip().startswith("["):
                return decode_json(payload)
            return None
        return decode_json(payload)
    return None


def _collect_keys(value: object, depth: int = 0) -> set[str]:
    if depth > _KEY_SEARCH_DEPTH:
        return set()

    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                keys.add(key)
            if isinstance(item, (Mapping, list)):
                keys |= _collect_keys(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (Mapping, list)):
                keys |= _collect_keys(item, depth + 1)
    return keys


def detect_source(
    payload: str | bytes | Mapping[str, Any],
) -> ReportSource:
    """Infer the report format from a payload.

    Non-JSON payloads are Markdown (which includes email).  JSON that matches
    neither HackerOne nor GitHub raises instead of being guessed at: feeding an
    advisory export to the Markdown adapter would treat raw JSON as the report
    body and silently produce a meaningless ``NormalizedReport``.
    """

    data = _payload_object(payload)
    if data is None:
        return ReportSource.MARKDOWN

    keys = _collect_keys(data)
    if keys & _GITHUB_MARKERS:
        return ReportSource.GITHUB_ADVISORY

    # HackerOne requires at least one unambiguous ("strong") marker,
    # or `team` / `weakness` combined with `vulnerability_information`
    # — preventing generic objects like ``{"team": {...}}`` from being
    # mistaken for a report.
    has_strong = bool(keys & _HACKERONE_STRONG_MARKERS)
    has_weak = bool(keys & _HACKERONE_WEAK_MARKERS)
    has_report_key = bool(keys & {"vulnerability_information"})
    if has_strong or (has_weak and has_report_key):
        return ReportSource.HACKERONE

    raise IntakeError(
        "could not determine the report format; pass --source "
        "markdown|email|hackerone|github"
    )
