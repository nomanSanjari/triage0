"""Sandbox orchestration: version text in, isolated container with that code out.

The order is always: resolve the tag, check it out into a disposable
worktree, start a container with that worktree mounted, run commands,
then remove the container and the worktree, even after errors and timeouts.

The container backend (DockerLocal) only knows how to start/exec/stop.
This module decides what happens and guarantees the cleanup.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from triage0.checkers.tools.git_tools import (
    TagResolution,
    TagResolutionStatus,
    list_tags,
    resolve_tag,
)
from triage0.core.evidence import Argv, CommandResult, DeterministicProof
from triage0.sandbox.providers.docker_local import SANDBOX_SRC, DockerLocal, ExecResult
from triage0.storage.repo_cache import RepoCache, cache_subpath, worktree

DEFAULT_IMAGE = "triage0-sandbox:dev"

# CommandResult needs an int exit code, but a command we killed on timeout has
# none. -9 is how a process killed by SIGKILL is conventionally reported; the
# real signal that it timed out is the `timed_out` flag.
TIMEOUT_EXIT_CODE = -9


class TagNotResolved(Exception):
    """The version text didn't map to exactly one tag.

    Callers should report CANT_CHECK with `resolution.reason`, never
    CONTRADICTED: we simply don't know which code the report is about.
    """

    def __init__(self, resolution: TagResolution) -> None:
        super().__init__(resolution.reason)
        self.resolution = resolution


def _to_command_result(result: ExecResult) -> CommandResult:
    """Translate the container backend's result into the shared evidence type."""
    return CommandResult(
        # An odd program path like "/usr/bin/" has an empty basename.
        tool=os.path.basename(result.argv[0]) or result.argv[0],
        argv=result.argv,
        exit_code=TIMEOUT_EXIT_CODE if result.exit_code is None else result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.timed_out,
        truncated=result.truncated,
    )


@dataclass(frozen=True)
class Session:
    """A running sandbox with one exact version of the code at /src.

    Implements the SandboxSession protocol (core/interfaces.py). `tree_hash`
    and `commit` identify the code; evidence records them with each command.
    After a command times out the container is gone, so later run()/seal()
    calls raise SandboxError instead of continuing.
    """

    tag: str
    tree_hash: str
    commit: str
    container_id: str
    provider: DockerLocal

    def run(self, argv: Argv, *, timeout_s: float | None = None) -> CommandResult:
        """Run one command (no shell) inside the sandbox."""
        return _to_command_result(self.provider.exec(self.container_id, argv, timeout_s))

    def seal(self, argv: Argv, *, timeout_s: float | None = None) -> DeterministicProof:
        """Run the decisive command and seal the result as proof.

        This does not compare two runs; checking that a result is repeatable
        belongs to the proof gate. A timed-out or truncated run is still
        sealed, with those flags set, so EvidenceResult can refuse to base a
        CONFIRMED/CONTRADICTED verdict on it.
        """
        return DeterministicProof.from_command_result(
            self.run(argv, timeout_s=timeout_s),
            git_tree_hash=self.tree_hash,
            commit=self.commit,
            workdir=SANDBOX_SRC,
        )


def project_from_url(url: str) -> str:
    """Repo name from a trusted URL: https://github.com/curl/curl.git -> curl."""
    return cache_subpath(url).name.removesuffix(".git")


class SandboxManager:
    """Ties the repo cache, tag resolution and container backend together.

    The provider is typed as DockerLocal for now; it becomes the
    SandboxProvider protocol once task 1.1 defines it.
    """

    def __init__(
        self, repo_cache: RepoCache, provider: DockerLocal, image: str = DEFAULT_IMAGE
    ) -> None:
        self.repo_cache = repo_cache
        self.provider = provider
        self.image = image

    @contextmanager
    def session(
        self, repo_url: str, version_text: str, project: str | None = None
    ) -> Iterator[Session]:
        """Clone/fetch `repo_url` on the host, then open a session for `version_text`.

        `project` is the name versions are anchored to ("curl 8.13.0");
        it defaults to the repo name from the URL.
        """
        bare_repo = self.repo_cache.ensure_clone(repo_url)
        name = project if project is not None else project_from_url(repo_url)
        with self.session_for_repo(bare_repo, name, version_text) as session:
            yield session

    @contextmanager
    def session_for_repo(
        self, bare_repo: Path, project: str, version_text: str
    ) -> Iterator[Session]:
        """Open a session on a repo already on disk (no network access).

        ExitStack runs cleanups in reverse order: the container is removed
        first, then the worktree it was using.
        """
        resolution = resolve_tag(version_text, project, list_tags(bare_repo))
        if resolution.status is not TagResolutionStatus.RESOLVED or resolution.tag is None:
            raise TagNotResolved(resolution)

        with ExitStack() as cleanup:
            wt = cleanup.enter_context(worktree(bare_repo, resolution.tag))
            container_id = self.provider.start(self.image, wt.path)
            cleanup.callback(self.provider.stop, container_id)
            yield Session(
                tag=wt.tag,
                tree_hash=wt.tree_hash,
                commit=wt.commit,
                container_id=container_id,
                provider=self.provider,
            )

    def run_once(self, repo_url: str, version_text: str, argv: Argv) -> CommandResult:
        """Start a sandbox for one version, run one command, clean everything up."""
        with self.session(repo_url, version_text) as session:
            return session.run(argv)
