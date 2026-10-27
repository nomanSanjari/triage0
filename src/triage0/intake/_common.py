"""Shared parsing helpers for report intake adapters.

Adapters receive untrusted input from bug-bounty exports, mail archives, and
pasted text, so every helper here is total: it either returns a normalized
value or raises :class:`IntakeError`, a ``ValueError`` subclass that the CLI can
report in one place.

Nothing in this module imports the core models; it is pure text handling so the
three adapters do not each re-implement decoding and timestamp parsing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from hashlib import sha256
from typing import Any

__all__ = [
    "IntakeError",
    "clean_value",
    "decode_json",
    "decode_text",
    "derive_report_id",
    "first_line",
    "first_raw_text",
    "first_text",
    "first_timestamp",
    "json_safe",
    "labelled_value",
    "normalize_newlines",
    "parse_timestamp",
]

# Derived ids stay readable and bounded: ``<prefix>-<12 hex characters>``.
_ID_HASH_LENGTH = 12


class IntakeError(ValueError):
    """Raised when an incoming payload cannot be normalized."""


def decode_text(payload: str | bytes) -> str:
    """Return ``payload`` as text, decoding UTF-8 and dropping any BOM.

    ``utf-8-sig`` is used rather than ``utf-8`` because exported JSON and email
    files frequently begin with a BOM, which would otherwise make the first
    JSON key or header name unparseable.
    """

    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IntakeError(
                f"payload is not valid UTF-8 (byte {exc.start}: {exc.reason})"
            ) from exc
    if isinstance(payload, str):
        return payload
    raise IntakeError(f"expected text or bytes, got {type(payload).__name__}")


def decode_json(payload: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON object from text, bytes, or an already-parsed mapping."""

    if isinstance(payload, Mapping):
        return dict(payload)
    try:
        data = json.loads(decode_text(payload))
    except json.JSONDecodeError as exc:
        raise IntakeError(
            f"payload is not valid JSON ({exc.msg} at line {exc.lineno}, "
            f"column {exc.colno})"
        ) from exc
    if isinstance(data, list):
        if len(data) == 0:
            raise IntakeError(
                "JSON array payload contains zero reports; "
                "expected exactly one report object"
            )
        if len(data) == 1:
            item = data[0]
            if isinstance(item, Mapping):
                return dict(item)
            raise IntakeError(
                "JSON array payload contains non-object item; "
                "expected exactly one report object"
            )
        raise IntakeError(
            f"JSON array payload contains {len(data)} reports; "
            "expected exactly one report object"
        )
    if not isinstance(data, dict):
        raise IntakeError(
            "expected a JSON object at the top level, got "
            f"{type(data).__name__}"
        )
    return data


def parse_timestamp(value: object) -> datetime | None:
    """Parse a timestamp into canonical UTC.

    Accepts ``datetime`` objects, epoch seconds, ISO 8601 (including a trailing
    ``Z``), and RFC 2822 dates as used by email ``Date`` headers.  Unparseable
    values return ``None`` instead of raising: a malformed date must not cause a
    real vulnerability description to be discarded.

    :class:`~triage0.core.models.NormalizedReport` requires timezone-aware
    datetimes, so naively parsed values are pinned to UTC rather than left to
    fail model validation later.  Aware values are converted to UTC too, so
    reports sourced from different timezones share one representation.
    """

    if value is None or isinstance(value, bool):
        return None

    parsed: datetime | None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parsed = _parse_iso(text)
        if parsed is None:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError):
                return None
    else:
        return None

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_iso(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    # Python 3.11+ accepts a trailing ``Z`` natively; this fallback also
    # tolerates the lowercase ``z`` that some exporters emit.
    if text.endswith(("Z", "z")):
        try:
            return datetime.fromisoformat(f"{text[:-1]}+00:00")
        except ValueError:
            return None
    return None


def derive_report_id(prefix: str, body: str) -> str:
    """Derive a deterministic id from the canonical report body.

    Re-ingesting the same report produces the same id, which lets downstream
    deduplication key on it instead of on a filename.  The body is newline
    normalized and trimmed first so the id agrees with
    :attr:`~triage0.core.models.NormalizedReport.content_hash`, whose model
    strips surrounding whitespace.
    """

    canonical = normalize_newlines(body).strip()
    digest = sha256(canonical.encode("utf-8")).hexdigest()[:_ID_HASH_LENGTH]
    return f"{prefix}-{digest}"


def normalize_newlines(text: str) -> str:
    """Convert CRLF/CR line endings to LF without touching other whitespace."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def json_safe(value: Any) -> Any:
    """Return ``value`` containing only JSON-serializable types.

    Adapters store source-specific extras in ``NormalizedReport.metadata``,
    which must survive a ``model_dump_json`` round-trip (``triage0 verify
    --format json``).  Unknown objects are stringified rather than dropped.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


# Emphasis characters wrapped around labels and values in hand-written reports.
_EMPHASIS = "*_`"


def clean_value(value: str) -> str:
    """Strip surrounding emphasis, backticks, and whitespace from a value."""

    return value.strip().strip(_EMPHASIS).strip()


@lru_cache(maxsize=None)
def _label_pattern(labels: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a ``Label: value`` matcher for a fixed label set.

    Longest labels are tried first so ``affected version`` wins over
    ``version``.  Bullets, bold labels, and bold values are all tolerated,
    because exported reports and hand-written text use every combination:

    ``- **Affected version:** **8.13.0**``
    """

    alternation = "|".join(
        re.escape(label) for label in sorted(labels, key=len, reverse=True)
    )
    return re.compile(
        rf"^[ \t]*(?:[-*+][ \t]+)?(?:\*\*|__)?[ \t]*(?:{alternation})"
        rf"[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*"
        rf"(?P<value>.+?)[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )


def labelled_value(labels: Sequence[str], text: str) -> str | None:
    """Return the first non-empty ``Label: value`` match in ``text``."""

    match = _label_pattern(tuple(labels)).search(text)
    if match is None:
        return None
    return clean_value(match.group("value")) or None


def first_line(text: str, *, limit: int = 120) -> str | None:
    """First non-blank line of ``text``, cleaned and clipped to ``limit``."""

    for line in text.splitlines():
        candidate = clean_value(line)
        if candidate:
            return candidate[:limit]
    return None


def first_text(
    data: Mapping[str, Any], keys: Sequence[str]
) -> str | None:
    """First non-blank string among ``keys``; numbers are stringified.

    Adapters pass a list of aliases because the same field is spelled
    ``ghsa_id`` in GitHub's REST payload and ``ghsaId`` in GraphQL, and
    HackerOne exports are similarly inconsistent.
    """

    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def first_raw_text(
    data: Mapping[str, Any], keys: Sequence[str]
) -> str | None:
    """First non-blank string among ``keys``, preserving surrounding text.

    Used for report bodies, which must not be reflowed or trimmed before the
    model records them.
    """

    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def first_timestamp(
    data: Mapping[str, Any], keys: Sequence[str]
) -> datetime | None:
    """First timestamp among ``keys``, normalized to UTC by ``parse_timestamp``."""

    for key in keys:
        parsed = parse_timestamp(data.get(key))
        if parsed is not None:
            return parsed
    return None
