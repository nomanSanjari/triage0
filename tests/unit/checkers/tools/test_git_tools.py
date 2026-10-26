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
    run_git,
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


@pytest.mark.parametrize(
    "text",
    [
        "server at 127.0.0.1 running 8.13.0",
        "connect to 10.0.0.1:8080 and 8.13.0",
        "gateway 192.168.1.1. Affected: 8.13.0",
    ],
)
def test_find_versions_fallback_ignores_ip_addresses(text: str) -> None:
    assert find_versions(text, "curl") == {(8, 13, 0)}


def test_find_versions_fallback_with_only_an_ip_address_finds_nothing() -> None:
    assert find_versions("crash when connecting to 127.0.0.1", "curl") == set()


def test_find_versions_ip_filter_only_applies_to_four_numbers_up_to_255() -> None:
    assert find_versions("build 1.2.3.400", "curl") == {(1, 2, 3, 400)}  # 400 is no IPv4 part
    assert find_versions("build 1.2.3", "curl") == {(1, 2, 3)}  # three numbers
    assert find_versions("curl 1.2.3.4", "curl") == {(1, 2, 3, 4)}  # named after the project


def test_resolve_tag_with_an_ip_address_in_the_text() -> None:
    result = resolve_tag("server at 127.0.0.1 running 8.13.0", "curl", CURL_TAGS)
    assert result.status is TagResolutionStatus.RESOLVED
    assert result.tag == "curl-8_13_0"


def test_find_versions_rejects_dev_builds() -> None:
    # 8.13.0-DEV means "built from git": any commit after 8.12.x up to the
    # 8.13.0 tag. We can't know which, so we must not pick a tag.
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


def test_list_tags_ignores_inherited_git_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An inherited GIT_DIR would silently point git at another repo.
    _git(tmp_path, "init", "-q")
    _git(
        tmp_path,
        "-c", "user.name=test", "-c", "user.email=test@example.com",
        "commit", "-q", "--allow-empty", "-m", "init",
    )
    _git(tmp_path, "tag", "curl-8_13_0")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "nowhere"))
    assert list_tags(tmp_path) == ["curl-8_13_0"]


# --- run_git hardening (real git, no network) ----------------------------------


@pytest.mark.parametrize("url_template", ["ext::sh -c touch% {marker}", "file://{repo}", "{repo}"])
def test_run_git_blocks_non_https_transports(tmp_path: Path, url_template: str) -> None:
    # Our URL check already rejects these; this proves git itself refuses too.
    # Plain git allows file:// and local paths, so those cases test our
    # hardening. Plain git already blocks ext:: by default; that case only
    # guards against a config that re-enables it.
    repo, marker = tmp_path / "repo", tmp_path / "pwned"
    repo.mkdir()
    _git(repo, "init", "-q")
    url = url_template.format(marker=marker, repo=repo)
    with pytest.raises(subprocess.CalledProcessError):
        run_git(["ls-remote", url])
    assert not marker.exists()


def test_run_git_ignores_personal_git_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A personal config could enable LFS or filters that run programs on checkout.
    personal = tmp_path / "gitconfig"
    personal.write_text("[user]\n\tname = leaked\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".gitconfig").write_text(personal.read_text())
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(personal))
    with pytest.raises(subprocess.CalledProcessError):  # exit 1 = "key not set"
        run_git(["config", "--get", "user.name"])
