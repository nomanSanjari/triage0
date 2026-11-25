"""Unit tests for rendering untrusted report text in a summary."""

from __future__ import annotations

import pytest

from triage0.checkers.tools.report_text import quote


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("curl_easy_init", "'curl_easy_init'"),
        # Line breaks, control characters and quotes must not break out of the
        # quoted span: a summary has to stay one plain line.
        ("a\nb", "'a\\nb'"),
        ("a\x1b[31mb", "'a\\x1b[31mb'"),
        ("it's", '"it\'s"'),
    ],
)
def test_quote_keeps_untrusted_text_on_one_escaped_line(
    text: str, expected: str
) -> None:
    assert quote(text) == expected


def test_quote_bounds_what_a_report_can_put_in_a_summary() -> None:
    assert quote("x" * 500) == repr("x" * 80)
