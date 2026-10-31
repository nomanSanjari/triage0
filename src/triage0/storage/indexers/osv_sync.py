"""Build both curl datasets from public sources; also the CLI entry point.

This module coordinates the refresh: it owns request pacing, the temporary
curl clone lifecycle, stale-file checks, and the write step. The heavy
lifting lives in focused submodules:

- :mod:`osv_advisories` -- OSV models, ``vuln.json`` parsing, version ranges.
- :mod:`osv_refs` -- tag indexing, evaluation-ref resolution, commit checks.
- :mod:`osv_origins` -- report-origin matching and flat record assembly.

Network access happens here, on the host, once per dataset refresh. The
committed JSON files are the snapshot.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from triage0.checkers.tools.git_tools import (
    TagResolutionStatus,
    resolve_tag,
    run_git,
)
from triage0.storage.indexers.h1_reports import (
    GIST_HOSTS,
    SLOP_GIST_RAW_URL,
    FetchPolicy,
    H1Error,
    build_fake_record,
    fetch_public_report,
    fetch_public_text,
    parse_slop_list,
    report_text,
    write_json_record,
)
from triage0.storage.indexers.osv_advisories import (
    OsvError,
    OsvVulnerability,
    cve_id,
    load_vuln_json,
    osv_git_commits,
)
from triage0.storage.indexers.osv_origins import (
    Origin,
    _origin_for,
    build_real_record,
)
from triage0.storage.indexers.osv_refs import (
    CURL_REPO_GIT_URL,
    TagRef,
    build_tag_index,
    format_version,
    resolve_ref,
    validate_curl_repo,
    verify_commit,
    version_to_tag,
)

# Only names this module actually exposes. Owning modules are the import
# path for everything else (osv_advisories, osv_refs, osv_origins).
__all__ = [
    "CURL_REPO_GIT_URL",
    "CLONE_TIMEOUT_SECONDS",
    "Origin",
    "OsvError",
    "OsvVulnerability",
    "SyncSummary",
    "TagRef",
    "build_real_record",
    "build_tag_index",
    "cve_id",
    "load_vuln_json",
    "main",
    "resolve_ref",
    "sync",
    "validate_curl_repo",
    "verify_commit",
    "version_to_tag",
]

CLONE_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class SyncSummary:
    fake: int
    real: int
    hackerone: int
    github_issue: int
    osv_advisory: int
    unresolved_refs: int
    unverified_commits: int


class _Pacer:
    """Serial request pacing. HackerOne returns 429 under concurrency."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay = delay_seconds
        self._first = True

    def wait(self) -> None:
        if self._first:
            self._first = False
            return
        if self._delay > 0:
            time.sleep(self._delay)


def _check_stale(directory: Path, expected: set[str]) -> None:
    """Refuse a full refresh that would leave files from an older snapshot."""

    if not directory.is_dir():
        return
    stale = sorted(
        path.name for path in directory.glob("*.json") if path.stem not in expected
    )
    if stale:
        raise OsvError(
            "dataset contains files this run would not write: " + ", ".join(stale)
        )


@contextmanager
def _curl_repo_context(curl_repo: Path | None) -> Iterator[Path]:
    """Reuse a supplied curl clone, or make a temporary bare partial clone."""

    if curl_repo is not None:
        validate_curl_repo(curl_repo)
        yield curl_repo
        return
    with tempfile.TemporaryDirectory(prefix="triage0-curl-") as tmp:
        repo = Path(tmp) / "curl.git"
        # Only refs and commits are needed for tag lookup and verification, so
        # a bare blob-less partial clone avoids downloading curl's full history.
        run_git(
            [
                "clone",
                "--bare",
                "--filter=blob:none",
                CURL_REPO_GIT_URL,
                str(repo),
            ],
            timeout=CLONE_TIMEOUT_SECONDS,
        )
        yield repo


def sync(
    *,
    out_dir: Path,
    slop_dir: Path | None,
    policy: FetchPolicy,
    curl_repo: Path | None = None,
    vuln_json_path: Path | None = None,
    only: str | None = None,
    dry_run: bool = False,
) -> SyncSummary:
    """Build both datasets; the summary is what the CLI prints."""

    with _curl_repo_context(curl_repo) as repo:
        return _sync_with_repo(
            out_dir=out_dir,
            slop_dir=slop_dir,
            policy=policy,
            curl_repo=repo,
            vuln_json_path=vuln_json_path,
            only=only,
            dry_run=dry_run,
        )


def _sync_with_repo(
    *,
    out_dir: Path,
    slop_dir: Path | None,
    policy: FetchPolicy,
    curl_repo: Path,
    vuln_json_path: Path | None,
    only: str | None,
    dry_run: bool,
) -> SyncSummary:
    index = build_tag_index(curl_repo)
    tags = [ref.tag for ref in index.values()]
    pacer = _Pacer(policy.delay_seconds)

    fake = 0
    fake_records: list[dict[str, Any]] = []
    fake_ids: set[str] = set()
    unverified = 0
    # --only is a targeted CVE refresh; rebuilding the slop set would make it
    # download 49 unrelated reports.
    if slop_dir is not None and only is None:
        gist = fetch_public_text(
            SLOP_GIST_RAW_URL, policy, allowed_hosts=GIST_HOSTS
        )
        report_ids = parse_slop_list(gist)
        for position, report_id in enumerate(report_ids, 1):
            pacer.wait()
            report = fetch_public_report(report_id, policy)
            resolution = resolve_tag(
                report_text(report.title, report.body), "curl", tags
            )
            tag: str | None = None
            commit: str | None = None
            version: str | None = None
            if (
                resolution.status is TagResolutionStatus.RESOLVED
                and resolution.version is not None
            ):
                tag_ref = version_to_tag(format_version(resolution.version), index)
                if tag_ref is not None:
                    tag, commit = tag_ref.tag, tag_ref.commit
                    version = format_version(resolution.version)
            fake_commit_verified: bool | None = None
            if commit is not None:
                fake_commit_verified = verify_commit(curl_repo, commit)
                if not fake_commit_verified:
                    unverified += 1
                    print(
                        f"  ! h1-{report.id}: selected commit {commit} not found "
                        "in the curl repository; keeping the resolved ref"
                    )
            record = build_fake_record(
                report,
                tag=tag,
                commit=commit,
                reported_version=version,
                commit_verified=fake_commit_verified,
            )
            fake_ids.add(str(record["id"]))
            fake_records.append(record)
            fake += 1
            print(
                f"[slop {position}/{len(report_ids)}] {record['id']} "
                f"ref={tag or 'none'}"
            )

    vulnerabilities = load_vuln_json(vuln_json_path, policy)
    if only is not None:
        vulnerabilities = [osv for osv in vulnerabilities if cve_id(osv) == only]
        if not vulnerabilities:
            raise OsvError(f"no advisory matches {only}")

    real = hackerone = github_issue = osv_advisory = 0
    unresolved = 0
    real_records: list[dict[str, Any]] = []
    real_ids: set[str] = set()
    for position, osv in enumerate(vulnerabilities, 1):
        cve = cve_id(osv)
        ref = resolve_ref(osv, index)
        if ref.commit is None:
            unresolved += 1
        origin = _origin_for(osv, policy, pacer.wait)
        if origin.source == "hackerone":
            hackerone += 1
        elif origin.source == "github_issue":
            github_issue += 1
        else:
            osv_advisory += 1

        commit_verified: bool | None = None
        # The selected commit is often also an OSV introduced hash; tracking
        # what was checked stops one missing hash being counted twice.
        checked: set[str] = set()
        if ref.commit is not None:
            commit_verified = verify_commit(curl_repo, ref.commit)
            checked.add(ref.commit)
            if not commit_verified:
                unverified += 1
                print(
                    f"  ! {cve}: selected commit {ref.commit} not found in the "
                    "curl repository; keeping the official hash"
                )
        for sha in osv_git_commits(osv):
            if sha in checked:
                continue
            checked.add(sha)
            if not verify_commit(curl_repo, sha):
                unverified += 1
                print(
                    f"  ! {cve}: OSV commit {sha} not found in the curl repository"
                )

        record = build_real_record(
            osv, ref, origin, commit_verified=commit_verified
        )
        real_ids.add(str(record["id"]))
        real_records.append(record)
        real += 1
        print(
            f"[cve {position}/{len(vulnerabilities)}] {cve} "
            f"source={origin.source} ref={ref.tag or ref.commit or 'none'}"
        )

    if not dry_run:
        # Stale files are rejected before the first write, so a refresh cannot
        # mix two snapshots by leaving old files behind. Each record is then
        # replaced atomically, but a failure midway through the loop can still
        # leave some records new and some old.
        if only is None:
            if slop_dir is not None:
                _check_stale(slop_dir, fake_ids)
            _check_stale(out_dir, real_ids)
        if slop_dir is not None:
            for record in fake_records:
                write_json_record(slop_dir, record)
        for record in real_records:
            write_json_record(out_dir, record)

    return SyncSummary(
        fake=fake,
        real=real,
        hackerone=hackerone,
        github_issue=github_issue,
        osv_advisory=osv_advisory,
        unresolved_refs=unresolved,
        unverified_commits=unverified,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the curl slop and CVE datasets from public sources."
    )
    parser.add_argument("--out", type=Path, default=Path("datasets/curl_cves_239"))
    parser.add_argument(
        "--slop-out", type=Path, default=Path("datasets/curl_slop_49")
    )
    parser.add_argument(
        "--skip-slop", action="store_true", help="only build the CVE dataset"
    )
    parser.add_argument(
        "--vuln-json", type=Path, default=None, help="use a local vuln.json snapshot"
    )
    parser.add_argument(
        "--curl-repo",
        type=Path,
        default=None,
        help=(
            "reuse this local curl clone instead of creating a temporary "
            "bare partial clone"
        ),
    )
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--only",
        default=None,
        help="rebuild one CVE id; the slop dataset is left untouched",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        # FetchPolicy enforces the flag bounds, so invalid --delay/--timeout/
        # --retries values exit 1 with a message instead of a traceback.
        policy = FetchPolicy(
            timeout=args.timeout, retries=args.retries, delay_seconds=args.delay
        )
        summary = sync(
            out_dir=args.out,
            slop_dir=None if args.skip_slop else args.slop_out,
            policy=policy,
            curl_repo=args.curl_repo,
            vuln_json_path=args.vuln_json,
            only=args.only,
            dry_run=args.dry_run,
        )
    except (
        ValidationError,
        H1Error,
        OsvError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        "built "
        f"{summary.fake} fake and {summary.real} real records "
        f"(hackerone={summary.hackerone}, github_issue={summary.github_issue}, "
        f"osv_advisory={summary.osv_advisory}, "
        f"unresolved_refs={summary.unresolved_refs}, "
        f"unverified_commits={summary.unverified_commits})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
