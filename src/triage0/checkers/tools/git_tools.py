"""Git helpers: tag listing and fuzzy tag resolution.

Bug report text is untrusted, attacker-controlled input. It is only ever
parsed here in Python, never passed to a command. The only tag that can
come out of resolve_tag() is one copied from the repository's own
`git tag --list`, so callers can safely pass it on to git.

Resolution is deliberately strict: when in doubt we refuse to pick a tag,
because checking the wrong code could turn a true claim into a false
CONTRADICTED. A refusal only ever leads to CANT_CHECK.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

GIT_TIMEOUT_SECONDS = 30

# Repos we check are untrusted. These settings stop git from running
# anything the repo or the user's personal config asks for: hooks, LFS
# downloads, credential prompts, or non-https transports like ext::.
_GIT_SAFE_CONFIG = [
    "-c", "core.hooksPath=/dev/null",
    "-c", "protocol.allow=never",
    "-c", "protocol.https.allow=always",
]
_GIT_SAFE_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,  # ignore ~/.gitconfig (LFS filters, aliases)
    "GIT_CONFIG_NOSYSTEM": "1",       # ignore /etc/gitconfig
    "GIT_TERMINAL_PROMPT": "0",       # fail instead of waiting for a password
    "GIT_LFS_SKIP_SMUDGE": "1",       # never download LFS files
}

# Two or more numbers joined by "." or "_" (curl tags use "_": curl-8_13_0).
_VERSION = r"\d+(?:[._]\d+)+"

# The version must not be glued to more text. A trailing "-rc1" or extra
# digits means a different build. curl's "-DEV" means built from git: any
# commit after the previous release up to this tag, so we can't pick one.
_VERSION_END = r"(?![-\w]|[._]\d)"

# A bare version in prose, not part of a longer token like "libcurl/8.1".
_BARE_VERSION_RE = re.compile(rf"(?<![\w./-])({_VERSION}){_VERSION_END}")

# Release tags we understand: optional name prefix + "-" or "_", optional
# "v", then a version. Anything else (rc tags, tags starting with "-")
# never matches, so it can never be selected.
_TAG_RE = re.compile(r"(?:[A-Za-z][A-Za-z0-9]*[-_])?v?(\d+(?:[._]\d+)+)")

# Project names come from trusted config (e.g. the repo URL), but we still
# keep them to a simple shape.
_PROJECT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

Version = tuple[int, ...]


class TagResolutionStatus(Enum):
    """Why a resolution did or did not produce a tag."""

    RESOLVED = "resolved"
    NO_VERSION = "no_version"
    AMBIGUOUS_VERSION = "ambiguous_version"
    NO_MATCHING_TAG = "no_matching_tag"
    MULTIPLE_MATCHING_TAGS = "multiple_matching_tags"


@dataclass(frozen=True)
class TagResolution:
    """Outcome of resolve_tag().

    `reason` is shown to maintainers to explain a CANT_CHECK, so it only
    contains numbers we parsed and tag names from git, never raw report text.
    """

    status: TagResolutionStatus
    tag: str | None
    version: Version | None
    reason: str


def _to_version(text: str) -> Version:
    return tuple(int(part) for part in re.split(r"[._]", text))


def _fmt(version: Version) -> str:
    return ".".join(str(part) for part in version)


def find_versions(text: str, project: str) -> set[Version]:
    """Return the release versions the report text mentions for `project`.

    Versions written right after the project name ("curl 8.13.0",
    "curl/8.13.0") win. Only if there are none do we fall back to every
    bare version in the text, since reports also mention versions of other
    software (OpenSSL/3.0.2, HTTP/1.1, IP addresses).
    """
    if not _PROJECT_RE.fullmatch(project):
        raise ValueError(f"invalid project name: {project!r}")
    anchored = re.compile(
        rf"(?<![\w-]){re.escape(project)}[ /-]v?({_VERSION}){_VERSION_END}",
        re.IGNORECASE,
    )
    found = {_to_version(m) for m in anchored.findall(text)}
    if found:
        return found
    return {_to_version(m) for m in _BARE_VERSION_RE.findall(text)}


def tag_version(tag: str) -> Version | None:
    """Return the version a release tag names, or None if it isn't one.

    "curl-8_13_0", "v8.13.0" and "8.13.0" all give (8, 13, 0).
    "curl-8_13_0-rc1" gives None: release candidates are different code.
    """
    match = _TAG_RE.fullmatch(tag)
    return _to_version(match.group(1)) if match else None


def resolve_tag(version_text: str, project: str, tags: Iterable[str]) -> TagResolution:
    """Map a free-text version like "curl 8.13.0" to exactly one real tag.

    Succeeds only when the text names exactly one version and exactly one
    tag has that version. Versions must match exactly: "8.13" does not
    match tag 8.13.0, because the tag list grows over time and a looser
    rule would make the same report resolve differently later.
    """
    versions = find_versions(version_text, project)
    if not versions:
        return TagResolution(
            TagResolutionStatus.NO_VERSION, None, None, "no release version found in the text"
        )
    if len(versions) > 1:
        listed = ", ".join(sorted(_fmt(v) for v in versions))
        return TagResolution(
            TagResolutionStatus.AMBIGUOUS_VERSION, None, None, f"text mentions several versions: {listed}"
        )

    (version,) = versions
    matches = sorted(tag for tag in tags if tag_version(tag) == version)
    if not matches:
        return TagResolution(
            TagResolutionStatus.NO_MATCHING_TAG, None, version, f"no release tag for version {_fmt(version)}"
        )
    if len(matches) > 1:
        return TagResolution(
            TagResolutionStatus.MULTIPLE_MATCHING_TAGS,
            None,
            version,
            f"several tags match version {_fmt(version)}: {', '.join(matches)}",
        )
    return TagResolution(
        TagResolutionStatus.RESOLVED, matches[0], version, f"resolved to tag {matches[0]}"
    )


def run_git(
    args: list[str], *, repo: Path | None = None, timeout: float = GIT_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run git with hardened settings. The single place all git calls go through.

    Any GIT_* variables inherited from the caller (e.g. GIT_DIR) are dropped
    so they can't silently point git at a different repo. Raises
    CalledProcessError on a non-zero exit and TimeoutExpired if it hangs.
    """
    command = ["git", *_GIT_SAFE_CONFIG]
    if repo is not None:
        command += ["-C", str(repo)]
    command += args
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(_GIT_SAFE_ENV)
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout,
        check=True,
    )


def list_tags(repo: Path) -> list[str]:
    """Return every tag name in `repo` (bare or not)."""
    return run_git(["tag", "--list"], repo=repo).stdout.splitlines()
