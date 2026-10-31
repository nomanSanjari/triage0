"""Tag indexing and evaluation-reference resolution.

This module maps release versions to tags and the commit each tag points at,
then picks the immutable commit to check a real advisory at. The commit is
what the baseline loader uses; the tag is kept for humans. When no tag exists
(curl 7.4 was never tagged), the OSV GIT ``introduced`` commit is used instead
and marked as such. A ``fixed`` version is never used as a ref.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from triage0.checkers.tools.git_tools import (
    Version,
    run_git,
    tag_version,
)
from triage0.storage.indexers.osv_advisories import (
    OsvError,
    OsvVulnerability,
    SHA,
    introduced_commits,
)

__all__ = [
    "CURL_REPO_GIT_URL",
    "RefResolution",
    "TagRef",
    "build_tag_index",
    "format_version",
    "parse_local_tag_refs",
    "parse_tag_refs",
    "resolve_ref",
    "validate_curl_repo",
    "verify_commit",
    "version_to_tag",
]

CURL_REPO_GIT_URL = "https://github.com/curl/curl.git"


@dataclass(frozen=True)
class TagRef:
    tag: str
    commit: str


@dataclass(frozen=True)
class RefResolution:
    tag: str | None
    commit: str | None
    reported_version: str
    ref_source: str


def format_version(version: Version) -> str:
    return ".".join(str(part) for part in version)


def _index_from_pairs(pairs: Iterable[tuple[str, str]]) -> dict[str, TagRef]:
    index: dict[str, TagRef] = {}
    for name, commit in pairs:
        version = tag_version(name)
        if version is None:
            # Release candidates and pre-releases are different code.
            continue
        key = format_version(version)
        existing = index.get(key)
        if existing is not None and existing.commit != commit:
            # Picking silently could check the wrong code and flip a verdict,
            # so an ambiguous version is refused instead of resolved.
            raise OsvError(f"several tags name version {key} with different commits")
        index[key] = TagRef(tag=name, commit=commit)
    return index


def parse_tag_refs(text: str) -> dict[str, TagRef]:
    """Parse ``git ls-remote --tags`` output into a version index."""

    direct: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, ref = line.partition("\t")
        if not ref.startswith("refs/tags/") or not SHA.fullmatch(sha):
            continue
        name = ref[len("refs/tags/"):]
        # Annotated tags appear twice: the tag object and its "^{}" peeled
        # commit. The peeled line is the commit we want to check out.
        if name.endswith("^{}"):
            peeled[name[:-3]] = sha
        else:
            direct[name] = sha
    return _index_from_pairs(
        (name, peeled.get(name, sha)) for name, sha in direct.items()
    )


def parse_local_tag_refs(text: str) -> dict[str, TagRef]:
    """Parse ``git for-each-ref`` output into a version index.

    Fields are name, object name, and the peeled commit (empty for a
    lightweight tag).
    """

    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 3:
            continue
        name, object_name, peeled = (field.strip() for field in fields)
        if not SHA.fullmatch(object_name):
            continue
        pairs.append((name, peeled if SHA.fullmatch(peeled) else object_name))
    return _index_from_pairs(pairs)


def build_tag_index(curl_repo: Path | None = None) -> dict[str, TagRef]:
    """Map release versions to tags and the commit each tag points at."""

    if curl_repo is not None:
        output = run_git(
            [
                "for-each-ref",
                "--format=%(refname:strip=2)\t%(objectname)\t%(*objectname)",
                "refs/tags",
            ],
            repo=curl_repo,
        ).stdout
        return parse_local_tag_refs(output)
    output = run_git(["ls-remote", "--tags", CURL_REPO_GIT_URL]).stdout
    return parse_tag_refs(output)


def version_to_tag(
    version_text: str, index: Mapping[str, TagRef]
) -> TagRef | None:
    """Resolve a version string only when it matches a release tag exactly."""

    version = tag_version(version_text.strip())
    if version is None:
        return None
    return index.get(format_version(version))


def resolve_ref(
    osv: OsvVulnerability, index: Mapping[str, TagRef]
) -> RefResolution:
    """Pick the immutable commit to check a real advisory at.

    ``last_affected`` is the only correct version. A fixed version is never a
    candidate, even when it is the last event in the range.

    The GIT ``introduced`` fallback is used only when the advisory names one
    unique commit. Several introduced commits mean the advisory spans multiple
    ranges, so the first one is not necessarily the code state at
    ``last_affected``; refusing is safer than checking the wrong commit.
    """

    last_affected = osv.database_specific.last_affected
    ref = version_to_tag(last_affected, index)
    if ref is not None:
        return RefResolution(
            tag=ref.tag,
            commit=ref.commit,
            reported_version=last_affected,
            ref_source="last_affected",
        )
    commits = introduced_commits(osv)
    if len(commits) == 1:
        return RefResolution(
            tag=None,
            commit=commits[0],
            reported_version=last_affected,
            ref_source="introduced_commit_fallback",
        )
    return RefResolution(
        tag=None,
        commit=None,
        reported_version=last_affected,
        ref_source="unresolved",
    )


def verify_commit(curl_repo: Path, sha: str) -> bool:
    """Return whether ``sha`` names a commit in the local curl checkout."""

    # Hash text comes from external advisory data, so reject anything that is
    # not a hex hash before it is placed into a git argv (safety rule 2).
    if not SHA.fullmatch(sha):
        return False
    try:
        run_git(["cat-file", "-e", f"{sha}^{{commit}}"], repo=curl_repo)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def validate_curl_repo(curl_repo: Path) -> None:
    """Fail clearly when a supplied --curl-repo is not a usable repository."""

    if not curl_repo.is_dir():
        raise OsvError(f"--curl-repo is not a directory: {curl_repo}")
    try:
        run_git(["rev-parse", "--git-dir"], repo=curl_repo)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise OsvError(
            f"--curl-repo is not a git repository: {curl_repo}"
        ) from None
