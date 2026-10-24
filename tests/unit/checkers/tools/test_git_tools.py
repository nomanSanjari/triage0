"""Tests for tag listing and fuzzy tag resolution."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from triage0.checkers.tools.git_tools import (
    TagResolutionStatus,
    find_versions,
    list_tags,
    resolve_tag,
    tag_version,
)

CURL_TAGS = ["curl-8_12_1", "curl-8_13_0", "curl-8_13_0-rc1", "curl-8_14_0", "curl-7_9_8"]


# --- tag_version -----------------------------------------------------------


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("curl-8_13_0", (8, 13, 0)),
        ("v8.13.0", (8, 13, 0)),
        ("8.13.0", (8, 13, 0)),
        ("curl_7_9_8", (7, 9, 8)),
    ],
)
def test_tag_version_understands_release_tags(tag: str, expected: tuple[int, ...]) -> None:
    assert tag_version(tag) == expected


@pytest.mark.parametrize(
    "tag",
    ["curl-8_13_0-rc1", "--upload-pack=x-8_13_0", "release", "v8", "8.13.0-DEV"],
)
def test_tag_version_rejects_non_release_tags(tag: str) -> None:
    assert tag_version(tag) is None


# --- find_versions ---------------------------------------------------------


def test_find_versions_prefers_project_anchored_version() -> None:
    curl_v = "curl 8.13.0 (x86_64-pc-linux-gnu) libcurl/8.13.0 OpenSSL/3.0.2 zlib/1.2.11 HTTP/1.1"
    assert find_versions(curl_v, "curl") == {(8, 13, 0)}


def test_find_versions_falls_back_to_bare_versions() -> None:
    assert find_versions("Affected version: 8.13.0.", "curl") == {(8, 13, 0)}


def test_find_versions_rejects_dev_builds() -> None:
    # 8.13.0-DEV is unreleased code from before 8.13.0, not the 8.13.0 tag.
    assert find_versions("curl 8.13.0-DEV", "curl") == set()


def test_find_versions_ignores_other_projects_named_like_us() -> None:
    # "name/x.y.z" belongs to that name (like OpenSSL/3.0.2), so it's skipped.
    assert find_versions("libcurl/8.13.0", "curl") == set()
    assert find_versions("libcurl/8.13.0 and curl 8.12.1", "curl") == {(8, 12, 1)}


def test_find_versions_rejects_bad_project_name() -> None:
    with pytest.raises(ValueError):
        find_versions("curl 8.13.0", "-curl")


# --- resolve_tag -----------------------------------------------------------


@pytest.mark.parametrize("text", ["curl 8.13.0", "curl/8.13.0", "CURL 8_13_0", "curl v8.13.0"])
def test_resolve_tag_finds_the_curl_tag(text: str) -> None:
    result = resolve_tag(text, "curl", CURL_TAGS)
    assert result.status is TagResolutionStatus.RESOLVED
    assert result.tag == "curl-8_13_0"
    assert result.version == (8, 13, 0)


def test_resolve_tag_needs_an_exact_version() -> None:
    result = resolve_tag("curl 8.13", "curl", CURL_TAGS)
    assert result.status is TagResolutionStatus.NO_MATCHING_TAG
    assert result.tag is None


def test_resolve_tag_reports_ambiguous_text() -> None:
    result = resolve_tag("works in curl 8.12.1, broken in curl 8.13.0", "curl", CURL_TAGS)
    assert result.status is TagResolutionStatus.AMBIGUOUS_VERSION
    assert "8.12.1" in result.reason and "8.13.0" in result.reason


def test_resolve_tag_reports_missing_version() -> None:
    result = resolve_tag("curl crashes sometimes", "curl", CURL_TAGS)
    assert result.status is TagResolutionStatus.NO_VERSION


def test_resolve_tag_refuses_when_several_tags_match() -> None:
    result = resolve_tag("curl 8.13.0", "curl", ["curl-8_13_0", "v8.13.0"])
    assert result.status is TagResolutionStatus.MULTIPLE_MATCHING_TAGS
    assert result.tag is None


def test_resolve_tag_output_is_always_a_known_tag() -> None:
    # Attacker-style text: whatever comes out must be from the tag list.
    evil = "curl --upload-pack=touch${IFS}/tmp/pwned 8.13.0; rm -rf /"
    result = resolve_tag(evil, "curl", CURL_TAGS)
    assert result.tag in CURL_TAGS


# --- list_tags (real git, no network) ---------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=30)


def test_list_tags_reads_a_real_repo(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(
        tmp_path,
        "-c", "user.name=test", "-c", "user.email=test@example.com",
        "commit", "-q", "--allow-empty", "-m", "init",
    )
    _git(tmp_path, "tag", "curl-8_13_0")
    _git(tmp_path, "tag", "curl-8_14_0")
    assert sorted(list_tags(tmp_path)) == ["curl-8_13_0", "curl-8_14_0"]


def test_list_tags_fails_loudly_outside_a_repo(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        list_tags(tmp_path / "missing")
