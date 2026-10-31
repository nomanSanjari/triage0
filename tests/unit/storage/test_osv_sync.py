"""Tests for dataset synchronization, the CLI, and repository context."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from triage0.checkers.tools.git_tools import run_git
from triage0.storage.indexers import osv_sync
from triage0.storage.indexers.h1_reports import (
    FetchPolicy,
    H1Error,
    H1Report,
)
from triage0.storage.indexers.osv_advisories import (
    OsvError,
    OsvVulnerability,
)
from triage0.storage.indexers.osv_origins import Origin
from triage0.storage.indexers.osv_refs import TagRef
from triage0.storage.indexers.osv_sync import (
    CURL_REPO_GIT_URL,
    SyncSummary,
    _curl_repo_context,
    main,
    sync,
)

FIX_SHA = "d7546809d5bd548b69ec49fca26565c8f6f0d344"
INTRO_SHA = "ce0881edee3c78609eae49665fb70264d8786d29"


def make_repo(path: Path) -> Path:
    """Initialize a real repository so --curl-repo validation accepts it."""

    run_git(["init", "--quiet"], repo=path)
    return path


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


def osv_without_issue() -> OsvVulnerability:
    payload = osv_payload()
    del payload["database_specific"]["issue"]
    return OsvVulnerability.model_validate(payload)


# --- sync ------------------------------------------------------------------


def test_sync_counts_unverified_fake_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = H1Report(
        id=2199174,
        title="Test",
        body="curl 8.4.0 is affected.",
        public=True,
        team_handle="curl",
    )
    index = {"8.4.0": TagRef("curl-8_4_0", "a" * 40)}
    monkeypatch.setattr(osv_sync, "build_tag_index", lambda _: index)
    monkeypatch.setattr(osv_sync, "fetch_public_text", lambda *a, **k: "")
    monkeypatch.setattr(osv_sync, "parse_slop_list", lambda _: [2199174])
    monkeypatch.setattr(osv_sync, "fetch_public_report", lambda *a, **k: report)
    monkeypatch.setattr(osv_sync, "verify_commit", lambda *a, **k: False)
    monkeypatch.setattr(osv_sync, "load_vuln_json", lambda *a, **k: [])

    summary = sync(
        out_dir=tmp_path,
        slop_dir=tmp_path,
        policy=FetchPolicy(),
        curl_repo=make_repo(tmp_path),
        dry_run=True,
    )

    assert summary.fake == 1
    assert summary.unverified_commits == 1


def test_main_reports_git_timeouts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(**kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="git", timeout=30)

    monkeypatch.setattr(osv_sync, "sync", boom)

    assert main(["--skip-slop"]) == 1
    assert "error:" in capsys.readouterr().err


def test_curl_repo_context_reuses_a_supplied_repo(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    with _curl_repo_context(repo) as supplied:
        assert supplied == repo


def test_curl_repo_context_rejects_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(OsvError, match="not a directory"):
        with _curl_repo_context(tmp_path / "missing"):
            pass


def test_curl_repo_context_rejects_a_file(tmp_path: Path) -> None:
    path = tmp_path / "curl.git"
    path.write_text("not a repo", encoding="utf-8")

    with pytest.raises(OsvError, match="not a directory"):
        with _curl_repo_context(path):
            pass


def test_curl_repo_context_rejects_a_directory_that_is_not_a_repo(
    tmp_path: Path,
) -> None:
    with pytest.raises(OsvError, match="not a git repository"):
        with _curl_repo_context(tmp_path):
            pass


def test_sync_rejects_a_shell_like_curl_repo_path(tmp_path: Path) -> None:
    marker = "triage0-pwned-marker"
    hostile = tmp_path / f"& touch {marker}"
    hostile.mkdir()

    with pytest.raises(OsvError, match="not a git repository"):
        sync(
            out_dir=tmp_path / "cves",
            slop_dir=None,
            policy=FetchPolicy(delay_seconds=0),
            curl_repo=hostile,
            dry_run=True,
        )

    assert not (Path.cwd() / marker).exists()
    assert not (tmp_path / marker).exists()


def test_curl_repo_context_clones_a_bare_partial_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run_git(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        Path(args[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(osv_sync, "run_git", fake_run_git)

    with _curl_repo_context(None) as repo:
        assert repo.is_dir()

    assert calls == [
        [
            "clone",
            "--bare",
            "--filter=blob:none",
            CURL_REPO_GIT_URL,
            str(repo),
        ]
    ]
    assert not repo.exists()


def test_sync_verifies_selected_and_osv_git_commits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = {"8.22.0": TagRef("curl-8_22_0", "a" * 40)}
    monkeypatch.setattr(osv_sync, "build_tag_index", lambda _: index)
    monkeypatch.setattr(
        osv_sync, "load_vuln_json", lambda *a, **k: [osv_without_issue()]
    )
    checked: list[str] = []

    def fake_verify(repo: Path, sha: str) -> bool:
        checked.append(sha)
        return True

    monkeypatch.setattr(osv_sync, "verify_commit", fake_verify)

    summary = sync(
        out_dir=tmp_path,
        slop_dir=None,
        policy=FetchPolicy(delay_seconds=0),
        curl_repo=make_repo(tmp_path),
        dry_run=True,
    )

    assert summary.real == 1
    assert checked == ["a" * 40, FIX_SHA, INTRO_SHA]


def test_sync_counts_each_unverified_commit_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = osv_payload()
    payload["database_specific"]["last_affected"] = "7.4"
    payload["affected"][0]["ranges"] = [
        {
            "type": "GIT",
            "events": [
                {"introduced": INTRO_SHA},
                {"fixed": FIX_SHA},
            ],
        }
    ]
    vuln = OsvVulnerability.model_validate(payload)
    monkeypatch.setattr(osv_sync, "build_tag_index", lambda _: {})
    monkeypatch.setattr(osv_sync, "load_vuln_json", lambda *a, **k: [vuln])
    monkeypatch.setattr(osv_sync, "verify_commit", lambda *a, **k: False)
    # The payload names a HackerOne issue, so stub the origin instead of
    # letting the test reach the network.
    monkeypatch.setattr(
        osv_sync,
        "_origin_for",
        lambda *args, **kwargs: Origin(
            source="osv_advisory",
            source_url="https://curl.se/docs/CVE-2026-92391.html",
            title="Test advisory",
            body="Test advisory body.",
            fallback_reason="no_original_report_link",
        ),
    )

    summary = sync(
        out_dir=tmp_path,
        slop_dir=None,
        policy=FetchPolicy(delay_seconds=0),
        curl_repo=make_repo(tmp_path),
        dry_run=True,
    )

    # The selected commit is the same introduced hash; it must count once.
    assert summary.unverified_commits == 2


def test_sync_writes_nothing_when_a_fetch_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    slop = tmp_path / "slop"
    out = tmp_path / "cves"
    monkeypatch.setattr(osv_sync, "build_tag_index", lambda _: {})
    monkeypatch.setattr(osv_sync, "fetch_public_text", lambda *a, **k: "")
    monkeypatch.setattr(osv_sync, "parse_slop_list", lambda _: [1, 2])
    calls = {"count": 0}

    def fake_fetch(report_id: int, policy: FetchPolicy) -> H1Report:
        calls["count"] += 1
        if calls["count"] == 2:
            raise H1Error("boom")
        return H1Report(
            id=report_id,
            title="t",
            body="b",
            public=True,
            team_handle="curl",
        )

    monkeypatch.setattr(osv_sync, "fetch_public_report", fake_fetch)
    monkeypatch.setattr(osv_sync, "load_vuln_json", lambda *a, **k: [])

    with pytest.raises(H1Error, match="boom"):
        sync(
            out_dir=out,
            slop_dir=slop,
            policy=FetchPolicy(delay_seconds=0),
            curl_repo=make_repo(tmp_path),
            dry_run=False,
        )

    assert list(slop.glob("*.json")) == []
    assert list(out.glob("*.json")) == []


def test_sync_rejects_stale_files_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "cves"
    out.mkdir()
    (out / "stale.json").write_text("{}", encoding="utf-8")
    index = {"8.22.0": TagRef("curl-8_22_0", "a" * 40)}
    monkeypatch.setattr(osv_sync, "build_tag_index", lambda _: index)
    monkeypatch.setattr(
        osv_sync, "load_vuln_json", lambda *a, **k: [osv_without_issue()]
    )
    monkeypatch.setattr(osv_sync, "verify_commit", lambda *a, **k: True)

    with pytest.raises(OsvError, match="would not write"):
        sync(
            out_dir=out,
            slop_dir=None,
            policy=FetchPolicy(delay_seconds=0),
            curl_repo=make_repo(tmp_path),
            dry_run=False,
        )

    assert not (out / "CVE-2026-92391.json").exists()
    assert (out / "stale.json").exists()


def test_sync_only_skips_the_stale_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "cves"
    out.mkdir()
    (out / "stale.json").write_text("{}", encoding="utf-8")
    index = {"8.22.0": TagRef("curl-8_22_0", "a" * 40)}
    monkeypatch.setattr(osv_sync, "build_tag_index", lambda _: index)
    monkeypatch.setattr(
        osv_sync, "load_vuln_json", lambda *a, **k: [osv_without_issue()]
    )
    monkeypatch.setattr(osv_sync, "verify_commit", lambda *a, **k: True)

    summary = sync(
        out_dir=out,
        slop_dir=None,
        policy=FetchPolicy(delay_seconds=0),
        curl_repo=make_repo(tmp_path),
        only="CVE-2026-92391",
        dry_run=False,
    )

    assert summary.real == 1
    assert (out / "CVE-2026-92391.json").exists()
    assert (out / "stale.json").exists()


def test_sync_only_skips_the_slop_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = {"8.22.0": TagRef("curl-8_22_0", "a" * 40)}
    monkeypatch.setattr(osv_sync, "build_tag_index", lambda _: index)
    monkeypatch.setattr(
        osv_sync, "load_vuln_json", lambda *a, **k: [osv_without_issue()]
    )
    monkeypatch.setattr(osv_sync, "verify_commit", lambda *a, **k: True)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("--only must not fetch slop reports")

    monkeypatch.setattr(osv_sync, "fetch_public_text", fail)
    monkeypatch.setattr(osv_sync, "fetch_public_report", fail)
    slop = tmp_path / "slop"

    summary = sync(
        out_dir=tmp_path / "cves",
        slop_dir=slop,
        policy=FetchPolicy(delay_seconds=0),
        curl_repo=make_repo(tmp_path),
        only="CVE-2026-92391",
        dry_run=False,
    )

    assert summary.fake == 0
    assert not slop.exists()


def test_sync_dry_run_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = {"8.22.0": TagRef("curl-8_22_0", "a" * 40)}
    monkeypatch.setattr(osv_sync, "build_tag_index", lambda _: index)
    monkeypatch.setattr(
        osv_sync, "load_vuln_json", lambda *a, **k: [osv_without_issue()]
    )
    monkeypatch.setattr(osv_sync, "verify_commit", lambda *a, **k: True)
    out = tmp_path / "cves"

    summary = sync(
        out_dir=out,
        slop_dir=None,
        policy=FetchPolicy(delay_seconds=0),
        curl_repo=make_repo(tmp_path),
        dry_run=True,
    )

    assert summary.real == 1
    assert not out.exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--delay", "-1"],
        ["--timeout", "0"],
        ["--retries", "-1"],
    ],
)
def test_main_rejects_an_invalid_fetch_policy(
    capsys: pytest.CaptureFixture[str], flags: list[str]
) -> None:
    assert main(flags) == 1
    assert "error:" in capsys.readouterr().err


def test_main_skip_slop_disables_the_slop_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_sync(**kwargs: object) -> SyncSummary:
        captured.update(kwargs)
        return SyncSummary(
            fake=0,
            real=0,
            hackerone=0,
            github_issue=0,
            osv_advisory=0,
            unresolved_refs=0,
            unverified_commits=0,
        )

    monkeypatch.setattr(osv_sync, "sync", fake_sync)

    assert main(["--skip-slop"]) == 0
    assert captured["slop_dir"] is None


def test_sync_warns_when_the_selected_commit_is_unverified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index = {"8.22.0": TagRef("curl-8_22_0", "a" * 40)}
    monkeypatch.setattr(osv_sync, "build_tag_index", lambda _: index)
    monkeypatch.setattr(
        osv_sync, "load_vuln_json", lambda *a, **k: [osv_without_issue()]
    )
    monkeypatch.setattr(osv_sync, "verify_commit", lambda *a, **k: False)
    out = tmp_path / "cves"

    summary = sync(
        out_dir=out,
        slop_dir=None,
        policy=FetchPolicy(delay_seconds=0),
        curl_repo=make_repo(tmp_path),
        dry_run=False,
    )

    assert summary.unverified_commits == 3
    assert f"selected commit {'a' * 40} not found" in capsys.readouterr().out
    record = json.loads((out / "CVE-2026-92391.json").read_text(encoding="utf-8"))
    assert record["commit"] == "a" * 40
    assert record["provenance"]["ref_verified"] is False


def test_failed_only_refresh_keeps_the_existing_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "cves"
    out.mkdir()
    existing = out / "CVE-2026-92391.json"
    original = '{"id": "CVE-2026-92391", "report": "old snapshot"}\n'
    existing.write_text(original, encoding="utf-8")
    index = {"8.22.0": TagRef("curl-8_22_0", "a" * 40)}
    monkeypatch.setattr(osv_sync, "build_tag_index", lambda _: index)
    monkeypatch.setattr(
        osv_sync, "load_vuln_json", lambda *a, **k: [osv_without_issue()]
    )

    def boom(*args: object, **kwargs: object) -> None:
        raise OsvError("origin fetch failed")

    monkeypatch.setattr(osv_sync, "_origin_for", boom)

    with pytest.raises(OsvError, match="origin fetch failed"):
        sync(
            out_dir=out,
            slop_dir=None,
            policy=FetchPolicy(delay_seconds=0),
            curl_repo=make_repo(tmp_path),
            only="CVE-2026-92391",
            dry_run=False,
        )

    assert existing.read_text(encoding="utf-8") == original


def test_all_exports_exist() -> None:
    # A name in __all__ that the module lacks breaks `from ... import *`.
    assert all(hasattr(osv_sync, name) for name in osv_sync.__all__)
