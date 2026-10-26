"""Local Docker backend: the only code that runs `docker` commands.

manager.py decides *what* happens (worktree, then container, then cleanup).
This module only knows *how* to start, use and remove one container.

Every container gets the full isolation set from build_run_args(): no
network, memory/CPU/pids limits, read-only filesystem, no capabilities,
and a hard lifetime. Never remove any of these, not even to test.

Note: this matches the planned SandboxProvider protocol (start/exec/stop)
by shape only; that protocol (core/interfaces.py, task 1.1) is still empty.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

DOCKER_TIMEOUT_SECONDS = 60
SANDBOX_LABEL = "triage0.sandbox=1"
SANDBOX_SRC = "/src"
MAX_OUTPUT_BYTES = 1024 * 1024
_READ_CHUNK = 64 * 1024

_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")


class SandboxError(Exception):
    """The container could not be started or used."""


@dataclass(frozen=True)
class SandboxLimits:
    """Resource limits for one container.

    Swap equals memory, so a container can't use extra swap. The lifetime
    is enforced inside the container (`sleep`), so it dies on its own even
    if our cleanup never runs. These belong in core/config.py once it exists.
    """

    memory: str = "1g"
    cpus: str = "1"
    pids: int = 256
    tmpfs_size: str = "64m"
    lifetime_seconds: int = 600
    exec_timeout_seconds: int = 60


@dataclass(frozen=True)
class ExecResult:
    """What one command inside the sandbox produced.

    `exit_code` is None when the command timed out. `truncated` means some
    output beyond MAX_OUTPUT_BYTES (per stream) was thrown away.
    """

    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


def build_run_args(image: str, src: Path, user: str, limits: SandboxLimits) -> list[str]:
    """Return the full `docker run` command for a sandbox container.

    Kept pure (no side effects) so tests can check every isolation flag.
    """
    if image.startswith("-") or not image:
        raise ValueError(f"invalid image name: {image!r}")
    if not src.is_absolute() or "," in str(src) or "=" in str(src):
        # --mount is a comma-separated key=value list; a path containing
        # "," or "=" could smuggle in extra mount options.
        raise ValueError(f"unsafe mount path: {src}")
    if not re.fullmatch(r"[1-9]\d*:\d+", user):
        raise ValueError("sandbox must run as a non-root uid:gid")
    return [
        "docker", "run", "--detach", "--rm", "--init",
        "--network=none",
        f"--memory={limits.memory}",
        f"--memory-swap={limits.memory}",
        f"--cpus={limits.cpus}",
        f"--pids-limit={limits.pids}",
        "--read-only",
        "--tmpfs", f"/tmp:rw,nosuid,nodev,size={limits.tmpfs_size}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--user={user}",
        f"--label={SANDBOX_LABEL}",
        "--mount", f"type=bind,src={src},dst={SANDBOX_SRC},readonly",
        f"--workdir={SANDBOX_SRC}",
        image,
        "sleep", str(limits.lifetime_seconds),
    ]


def _host_user() -> str:
    """uid:gid of this process, so the container can read our private worktree."""
    return f"{os.getuid()}:{os.getgid()}"


def _drain(stream: IO[bytes], keep: bytearray, truncated: list[bool]) -> None:
    """Read a stream to the end, keeping at most MAX_OUTPUT_BYTES of it.

    We keep reading after the limit (and throw it away) so the command
    doesn't block on a full pipe; the timeout still stops it.
    """
    while chunk := stream.read(_READ_CHUNK):
        room = MAX_OUTPUT_BYTES - len(keep)
        if len(chunk) > room:
            truncated[0] = True
        keep += chunk[: max(room, 0)]


def run_capped(command: Sequence[str], timeout: float) -> tuple[int | None, bytes, bytes, bool]:
    """Run a command with a timeout and bounded memory use.

    Returns (exit_code or None on timeout, stdout, stderr, truncated).
    subprocess.run(capture_output=True) would buffer unlimited output in
    our own process, so a sandboxed `cat /dev/urandom` could exhaust the
    host's memory. Here each stream keeps only MAX_OUTPUT_BYTES.
    """
    out, err = bytearray(), bytearray()
    truncated = [False]
    with subprocess.Popen(
        list(command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        assert proc.stdout is not None and proc.stderr is not None
        readers = [
            threading.Thread(target=_drain, args=(proc.stdout, out, truncated), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, err, truncated), daemon=True),
        ]
        for reader in readers:
            reader.start()
        try:
            exit_code: int | None = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            exit_code = None
        for reader in readers:
            reader.join(timeout=5)
    return exit_code, bytes(out), bytes(err), truncated[0]


def _check_container_id(container_id: str) -> str:
    if not _CONTAINER_ID.fullmatch(container_id):
        raise SandboxError(f"unexpected container id: {container_id!r}")
    return container_id


class DockerLocal:
    """Starts, runs commands in, and removes sandbox containers via the docker CLI."""

    def __init__(self, limits: SandboxLimits | None = None) -> None:
        self.limits = limits or SandboxLimits()

    def start(self, image: str, src: Path) -> str:
        """Start an idle sandbox container with `src` mounted read-only at /src.

        Returns the container id. The container runs `sleep <lifetime>` and
        removes itself when that ends, as a backstop if stop() never runs.
        """
        user = _host_user()
        if user.startswith("0:"):
            raise SandboxError("refusing to start a sandbox as root; run triage0 as a normal user")
        command = build_run_args(image, src, user, self.limits)
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=DOCKER_TIMEOUT_SECONDS
        )
        if result.returncode != 0:
            raise SandboxError(f"docker run failed: {result.stderr.strip()}")
        return _check_container_id(result.stdout.strip())

    def exec(self, container_id: str, argv: Sequence[str], timeout: float | None = None) -> ExecResult:
        """Run `argv` inside the container (no shell) and collect its output.

        On timeout the whole container is removed: a runaway process would
        otherwise keep running inside it. Later exec() calls will then fail.
        """
        if isinstance(argv, (str, bytes)):
            # *"ls -la" would be split into single characters, not words.
            raise TypeError("argv must be a list of arguments, not a single string")
        if not argv:
            raise ValueError("argv must not be empty")
        _check_container_id(container_id)
        limit = timeout if timeout is not None else self.limits.exec_timeout_seconds
        command = ["docker", "exec", container_id, *argv]
        exit_code, out, err, truncated = run_capped(command, limit)
        timed_out = exit_code is None
        if timed_out:
            self.stop(container_id)
        elif exit_code != 0 and err.startswith(b"Error response from daemon:"):
            # Docker itself failed (e.g. container gone). Its exit code must
            # not be mistaken for the command's exit code in evidence.
            raise SandboxError(f"docker exec failed: {err.decode('utf-8', errors='replace').strip()}")
        return ExecResult(
            argv=tuple(argv),
            exit_code=exit_code,
            stdout=out.decode("utf-8", errors="replace"),
            stderr=err.decode("utf-8", errors="replace"),
            timed_out=timed_out,
            truncated=truncated,
        )

    def stop(self, container_id: str) -> None:
        """Force-remove the container. Removing one that's already gone is fine."""
        _check_container_id(container_id)
        result = subprocess.run(
            ["docker", "rm", "--force", container_id],
            capture_output=True, text=True, timeout=DOCKER_TIMEOUT_SECONDS,
        )
        if result.returncode != 0 and "No such container" not in result.stderr:
            raise SandboxError(f"docker rm failed: {result.stderr.strip()}")


def list_sandbox_containers() -> list[str]:
    """Ids of all containers we started that still exist (for leak checks)."""
    result = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--no-trunc", f"--filter=label={SANDBOX_LABEL}"],
        capture_output=True, text=True, timeout=DOCKER_TIMEOUT_SECONDS, check=True,
    )
    return result.stdout.split()
