"""Tests for tag indexing, ref resolution, and commit verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from triage0.checkers.tools.git_tools import run_git
from triage0.storage.indexers import osv_refs
from triage0.storage.indexers.osv_advisories import (
    OsvError,
    OsvVulnerability,
)
from triage0.storage.indexers.osv_refs import (
    RefResolution,
    TagRef,
    build_tag_index,
    parse_local_tag_refs,
    parse_tag_refs,
    resolve_ref,
    verify_commit,
    version_to_tag,
)

FIX_SHA = "d7546809d5bd548b69ec49fca26565c8f6f0d344"
INTRO_SHA = "ce0881edee3c78609eae49665fb70264d8786d29"


def osv_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": "1.5.0",
        "id": "CURL-CVE-2026-92391",
        "aliases": ["CVE-2026-92391"],
        "summary": "curl_mime_free stack overflow by recursion",
        "details": "A deeply nested multipart MIME structure crashes on cleanup.",
        "published": "2026-10-14T08:00:00.00Z",
        "modified": "2026-10-24T23:17:46.00Z",
        "database_specific": {
            "package": "curl",
            "affects": "lib",
            "URL": "https://curl.se/docs/CVE-2026-92391.json",
            "www": "https://curl.se/docs/CVE-2026-92391.html",
            "issue": "https://hackerone.com/reports/4024222",
            "CWE": {"id": "CWE-674", "desc": "Uncontrolled Recursion"},
            "last_affected": "8.22.0",
            "severity": "Low",
        },
        "affected": [
            {
                "ranges": [
                    {
                        "type": "SEMVER",
                        "events": [
                            {"introduced": "7.56.0"},
                            {"fixed": "8.14.3"},
                            {"introduced": "8.15.0"},
                            {"fixed": "8.16.2"},
                            {"introduced": "8.17.0"},
                            {"fixed": "8.20.2"},
                            {"introduced": "8.21.0"},
                            {"fixed": "8.23.0"},
                        ],
                    },
                    {
                        "type": "GIT",
                        "repo": "https://github.com/curl/curl.git",
                        "events": [
                            {"introduced": INTRO_SHA},
                            {"fixed": FIX_SHA},
                        ],
                    },
                ],
                "versions": ["8.22.0"],
            }
        ],
    }
    data.update(overrides)
    return data


def osv(**overrides: Any) -> OsvVulnerability:
    return OsvVulnerability.model_validate(osv_payload(**overrides))


def index_with(*versions: tuple[str, str, str]) -> dict[str, TagRef]:
    return {
        version: TagRef(tag=tag, commit=commit)
        for version, tag, commit in versions
    }


# --- tag index -------------------------------------------------------------


def test_parse_tag_refs_prefers_the_peeled_commit() -> None:
    text = "\n".join(
        [
            f"{'1' * 40}\trefs/tags/curl-8_22_0",
            f"{'2' * 40}\trefs/tags/curl-8_22_0^{{}}",
            f"{'3' * 40}\trefs/tags/curl-8_21_0",
        ]
    )
    index = parse_tag_refs(text)

    assert index["8.22.0"] == TagRef("curl-8_22_0", "2" * 40)
    assert index["8.21.0"] == TagRef("curl-8_21_0", "3" * 40)


def test_parse_tag_refs_ignores_release_candidates() -> None:
    text = f"{'4' * 40}\trefs/tags/curl-8_23_0-rc1"
    assert parse_tag_refs(text) == {}


def test_parse_local_tag_refs_handles_both_tag_shapes() -> None:
    text = "\n".join(
        [
            f"curl-8_22_0\t{'5' * 40}\t{'6' * 40}",
            f"curl-8_21_0\t{'7' * 40}\t",
        ]
    )
    index = parse_local_tag_refs(text)

    assert index["8.22.0"].commit == "6" * 40
    assert index["8.21.0"].commit == "7" * 40


def test_parse_local_tag_refs_rejects_two_commits_for_one_version() -> None:
    text = "\n".join(
        [
            f"curl-8_22_0\t{'5' * 40}\t",
            f"curl-8.22.0\t{'6' * 40}\t",
        ]
    )

    with pytest.raises(OsvError, match="several tags name version 8.22.0"):
        parse_local_tag_refs(text)


def test_build_tag_index_reads_a_local_repository(tmp_path: Path) -> None:
    run_git(["init", "--quiet"], repo=tmp_path)
    run_git(
        [
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "init",
        ],
        repo=tmp_path,
    )
    run_git(["tag", "curl-8_22_0"], repo=tmp_path)
    sha = run_git(["rev-parse", "HEAD"], repo=tmp_path).stdout.strip()

    index = build_tag_index(tmp_path)
    assert index["8.22.0"] == TagRef("curl-8_22_0", sha)


def test_build_tag_index_resolves_an_annotated_tag(tmp_path: Path) -> None:
    run_git(["init", "--quiet"], repo=tmp_path)
    run_git(
        [
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "init",
        ],
        repo=tmp_path,
    )
    run_git(
        [
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "tag",
            "-a",
            "curl-8_22_0",
            "-m",
            "release",
        ],
        repo=tmp_path,
    )
    sha = run_git(["rev-parse", "HEAD"], repo=tmp_path).stdout.strip()
    tag_object = run_git(["rev-parse", "curl-8_22_0"], repo=tmp_path).stdout.strip()

    index = build_tag_index(tmp_path)
    # The index must point at the commit, not the annotated tag object.
    assert tag_object != sha
    assert index["8.22.0"] == TagRef("curl-8_22_0", sha)


def test_version_to_tag_requires_an_exact_release() -> None:
    index = index_with(("8.22.0", "curl-8_22_0", "a" * 40))
    assert version_to_tag("8.22.0", index) == TagRef("curl-8_22_0", "a" * 40)
    assert version_to_tag("7.4", index) is None
    assert version_to_tag("8.22", index) is None


# --- ref resolution --------------------------------------------------------


def test_resolve_ref_uses_last_affected_never_fixed() -> None:
    index = index_with(
        ("8.22.0", "curl-8_22_0", "a" * 40),
        ("8.23.0", "curl-8_23_0", "b" * 40),
    )
    ref = resolve_ref(osv(), index)

    assert ref == RefResolution(
        tag="curl-8_22_0",
        commit="a" * 40,
        reported_version="8.22.0",
        ref_source="last_affected",
    )
    assert ref.commit != "b" * 40


def test_resolve_ref_does_not_fall_back_to_the_fixed_version() -> None:
    index = index_with(("8.23.0", "curl-8_23_0", "b" * 40))
    payload = osv_payload()
    payload["affected"][0]["ranges"] = payload["affected"][0]["ranges"][:1]

    ref = resolve_ref(OsvVulnerability.model_validate(payload), index)
    assert ref.tag is None
    assert ref.commit is None
    assert ref.ref_source == "unresolved"


def test_resolve_ref_uses_introduced_commit_when_tag_is_missing() -> None:
    payload = osv_payload()
    payload["database_specific"]["last_affected"] = "7.4"

    ref = resolve_ref(OsvVulnerability.model_validate(payload), {})
    assert ref == RefResolution(
        tag=None,
        commit=INTRO_SHA,
        reported_version="7.4",
        ref_source="introduced_commit_fallback",
    )


def test_resolve_ref_is_unresolved_without_tag_or_commit() -> None:
    payload = osv_payload()
    payload["database_specific"]["last_affected"] = "7.4"
    payload["affected"][0]["ranges"] = payload["affected"][0]["ranges"][:1]

    ref = resolve_ref(OsvVulnerability.model_validate(payload), {})
    assert ref.commit is None
    assert ref.ref_source == "unresolved"


def test_resolve_ref_refuses_to_guess_between_introduced_commits() -> None:
    payload = osv_payload()
    payload["database_specific"]["last_affected"] = "7.4"
    payload["affected"][0]["ranges"] = [
        {
            "type": "GIT",
            "events": [
                {"introduced": INTRO_SHA},
                {"fixed": FIX_SHA},
            ],
        },
        {
            "type": "GIT",
            "events": [
                {"introduced": "b" * 40},
                {"fixed": "c" * 40},
            ],
        },
    ]

    ref = resolve_ref(OsvVulnerability.model_validate(payload), {})
    assert ref.commit is None
    assert ref.ref_source == "unresolved"


# --- commit verification ---------------------------------------------------


def test_verify_commit_checks_the_local_repository(tmp_path: Path) -> None:
    run_git(["init", "--quiet"], repo=tmp_path)
    run_git(
        [
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "init",
        ],
        repo=tmp_path,
    )
    sha = run_git(["rev-parse", "HEAD"], repo=tmp_path).stdout.strip()

    assert verify_commit(tmp_path, sha) is True
    assert verify_commit(tmp_path, "0" * 40) is False


def test_verify_commit_rejects_non_hashes_without_running_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(osv_refs, "run_git", lambda args, **kwargs: calls.append(args))

    assert verify_commit(tmp_path, "--upload-pack=evil") is False
    assert calls == []
