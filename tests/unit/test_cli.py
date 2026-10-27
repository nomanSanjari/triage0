"""Unit tests for the Typer CLI scaffold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from triage0 import __version__
from triage0.cli.main import app

runner = CliRunner()

HACKERONE_PAYLOAD: dict[str, Any] = {
    "id": 42,
    "title": "Heap overflow in curl_easy_escape",
    "vulnerability_information": "Affected version: 8.13.0\n\nDetails.",
    "team": {"handle": "curl"},
}

GITHUB_PAYLOAD: dict[str, Any] = {
    "ghsa_id": "GHSA-1111-2222-3333",
    "summary": "Out-of-bounds read",
    "description": "Affected version: 8.12.0\n\nDetails.",
    "html_url": "https://github.com/curl/curl/security/advisories/x",
    "vulnerabilities": [],
}

EMAIL_REPORT = """From: Researcher <researcher@example.com>
To: security@example.org
Subject: Heap overflow in curl_easy_escape
Date: Mon, 5 Oct 2026 14:30:00 +0000
Content-Type: text/plain; charset="utf-8"

Project: curl
Affected version: 8.13.0

Details.
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("verify", "eval", "index"):
        assert command in result.stdout


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_verify_markdown_terminal(tmp_path: Path) -> None:
    report = write(
        tmp_path,
        "report.md",
        "# Title\n\nProject: curl\n\nBody text\n",
    )

    result = runner.invoke(app, ["verify", "--report", str(report)])

    assert result.exit_code == 0
    assert "source:  markdown" in result.stdout
    assert "project: curl" in result.stdout
    assert "not wired yet" in result.stderr


def test_verify_json_output(tmp_path: Path) -> None:
    report = write(
        tmp_path,
        "report.md",
        "# Title\n\nProject: curl\n\nBody text\n",
    )

    result = runner.invoke(
        app, ["verify", "--report", str(report), "--format", "json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["source"] == "markdown"
    assert payload["project"] == "curl"
    assert payload["title"] == "Title"
    assert payload["report_id"].startswith("md-")


def test_verify_includes_repo_in_summary(tmp_path: Path) -> None:
    report = write(tmp_path, "report.md", "# Title\n\nBody\n")

    result = runner.invoke(
        app,
        [
            "verify",
            "--report",
            str(report),
            "--repo",
            "https://github.com/curl/curl.git",
        ],
    )

    assert result.exit_code == 0
    assert "repo:    https://github.com/curl/curl.git" in result.stdout


def test_verify_detects_hackerone(tmp_path: Path) -> None:
    report = write(
        tmp_path, "report.json", json.dumps(HACKERONE_PAYLOAD)
    )

    result = runner.invoke(
        app, ["verify", "--report", str(report), "--format", "json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["source"] == "hackerone"
    assert payload["report_id"] == "h1-42"


def test_verify_detects_github(tmp_path: Path) -> None:
    report = write(
        tmp_path, "report.json", json.dumps(GITHUB_PAYLOAD)
    )

    result = runner.invoke(
        app, ["verify", "--report", str(report), "--format", "json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["source"] == "github_advisory"
    assert payload["report_id"] == "GHSA-1111-2222-3333"


def test_verify_detects_email(tmp_path: Path) -> None:
    report = write(tmp_path, "report.eml", EMAIL_REPORT)

    result = runner.invoke(
        app, ["verify", "--report", str(report), "--format", "json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["source"] == "email"
    assert payload["title"] == "Heap overflow in curl_easy_escape"


def test_source_override_accepts_undetectable_payload(
    tmp_path: Path,
) -> None:
    payload = {"id": 7, "title": "Report", "body": "Body text"}
    report = write(tmp_path, "report.json", json.dumps(payload))

    auto = runner.invoke(app, ["verify", "--report", str(report)])
    forced = runner.invoke(
        app,
        [
            "verify",
            "--report",
            str(report),
            "--source",
            "hackerone",
            "--format",
            "json",
        ],
    )

    assert auto.exit_code == 1
    assert "could not determine" in auto.stderr
    assert forced.exit_code == 0
    assert json.loads(forced.stdout)["source"] == "hackerone"


def test_undetectable_json_exits_one(tmp_path: Path) -> None:
    report = write(tmp_path, "report.json", '{"foo": "bar"}')

    result = runner.invoke(app, ["verify", "--report", str(report)])

    assert result.exit_code == 1
    assert "could not determine the report format" in result.stderr


def test_invalid_json_exits_one(tmp_path: Path) -> None:
    report = write(tmp_path, "report.json", "{not json}")

    result = runner.invoke(app, ["verify", "--report", str(report)])

    assert result.exit_code == 1
    assert "not valid JSON" in result.stderr


def test_blank_report_exits_one(tmp_path: Path) -> None:
    report = write(tmp_path, "report.md", "   \n\n\t\n")

    result = runner.invoke(app, ["verify", "--report", str(report)])

    assert result.exit_code == 1
    assert "no report text" in result.stderr


def test_missing_report_option_exits_two() -> None:
    result = runner.invoke(app, ["verify"])

    assert result.exit_code == 2


def test_nonexistent_report_exits_two(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["verify", "--report", str(tmp_path / "missing.md")]
    )

    assert result.exit_code == 2


def test_out_writes_normalized_report(tmp_path: Path) -> None:
    report = write(tmp_path, "report.md", "# Title\n\nBody\n")
    out = tmp_path / "nested" / "normalized.json"

    result = runner.invoke(
        app,
        [
            "verify",
            "--report",
            str(report),
            "--format",
            "json",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["title"] == "Title"
    assert "Wrote normalized report" in result.stderr


def test_eval_requires_a_suite() -> None:
    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 2


def test_eval_is_explicitly_unimplemented() -> None:
    result = runner.invoke(
        app, ["eval", "--suite", "curl", "--include-baseline"]
    )

    assert result.exit_code == 2
    assert "not implemented yet" in result.stderr
    assert "suite=curl" in result.stderr
    assert "baseline=on" in result.stderr


def test_index_is_explicitly_unimplemented() -> None:
    result = runner.invoke(app, ["index", "--osv"])

    assert result.exit_code == 2
    assert "not implemented yet" in result.stderr
    assert "osv" in result.stderr


def test_json_array_raises_explicitly(tmp_path: Path) -> None:
    """Multi-element arrays must not fall through to Markdown silently."""

    report = write(tmp_path, "array.json", '[{"id":1}, {"id":2}]')

    result = runner.invoke(app, ["verify", "--report", str(report)])

    assert result.exit_code == 1
    assert "JSON array payload contains 2 reports" in result.stderr


def test_single_element_json_array_is_normalized(tmp_path: Path) -> None:
    """Exactly-one-element arrays are unambiguous and should work."""

    report = write(
        tmp_path,
        "single.json",
        '[{"id":42,"title":"Bug","vulnerability_information":"body"}]',
    )

    result = runner.invoke(
        app,
        ["verify", "--report", str(report), "--format", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["report_id"] == "h1-42"


def test_index_defaults_to_all_indexes() -> None:
    result = runner.invoke(app, ["index"])

    assert result.exit_code == 2
    assert "(all)" in result.stderr


def test_weak_hackerone_markers_without_report_key_are_rejected(
    tmp_path: Path,
) -> None:
    """``{team: ..., description: ...}`` must not be mistaken for H1."""

    report = write(
        tmp_path,
        "fake.json",
        '{"team": {"handle": "security"}, '
        '"description": "internal notes, not a report"}',
    )

    result = runner.invoke(app, ["verify", "--report", str(report)])

    assert result.exit_code == 1
    assert "could not determine the report format" in result.stderr


def test_email_attachments_are_excluded_from_body(tmp_path: Path) -> None:
    """MIME attachments must not leak into the normalized body."""

    eml = (
        "From: test@example.org\n"
        "To: security@example.org\n"
        "Subject: Attachment test\n"
        "Content-Type: multipart/mixed; boundary=sep\n\n"
        "--sep\n"
        "Content-Type: text/plain\n\n"
        "Main report body.\n"
        "--sep\n"
        'Content-Type: text/plain; name="patch.txt"\n'
        'Content-Disposition: attachment; filename="patch.txt"\n\n'
        "SECRET PATCH ATTACHMENT\n"
        "--sep--\n"
    )
    report = write(tmp_path, "with_attach.eml", eml)

    result = runner.invoke(
        app,
        ["verify", "--report", str(report), "--format", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["body"] == "Main report body."
    assert payload["metadata"]["attachments"] == ["patch.txt"]
