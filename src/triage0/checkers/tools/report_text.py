"""Rendering of report-supplied text for evidence summaries.

Summaries are shown in terminals and rendered as Markdown, so untrusted text
is escaped and bounded before it is interpolated. This is display only:
nothing here is compared, parsed or executed.
"""

from __future__ import annotations

__all__ = ["quote"]

_MAX_CHARS = 80


def quote(text: str) -> str:
    """Render untrusted text for a summary: escaped, bounded and one line."""

    return repr(text[:_MAX_CHARS])
