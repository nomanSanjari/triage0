"""Tests for docker_local that don't need Docker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from triage0.sandbox.providers.docker_local import (
    MAX_OUTPUT_BYTES,
    SANDBOX_LABEL,
    DockerLocal,
    SandboxLimits,
    build_run_args,
    run_capped,
)

SRC = Path("/tmp/triage0-wt-abc123")


def _args(**overrides: object) -> list[str]:
    params: dict[str, object] = {
        "image": "triage0-sandbox:dev", "src": SRC, "user": "1000:1000", "limits": SandboxLimits()
    }
    params.update(overrides)
    return build_run_args(**params)  # type: ignore[arg-type]


# --- build_run_args: guards CLAUDE.md rule 3 -----------------------------------


@pytest.mark.parametrize(
    "flag",
    [
        "--network=none",
        "--memory=1g",
        "--memory-swap=1g",
        "--cpus=1",
        "--pids-limit=256",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--rm",
        f"--label={SANDBOX_LABEL}",
    ],
)
def test_run_args_always_include_isolation_flag(flag: str) -> None:
    assert flag in _args()


def test_run_args_mount_worktree_read_only_and_expire() -> None:
    args = _args(limits=SandboxLimits(lifetime_seconds=42))
    assert f"type=bind,src={SRC},dst=/src,readonly" in args
    assert args[-3:] == ["triage0-sandbox:dev", "sleep", "42"]


def test_run_args_network_flag_cannot_be_overridden_by_later_flags() -> None:
    args = _args()
    assert [a for a in args if a.startswith("--network")] == ["--network=none"]


@pytest.mark.parametrize(
    "src",
    [Path("relative/path"), Path("/tmp/a,dst=/etc"), Path("/tmp/a=b")],
)
def test_run_args_reject_unsafe_mount_paths(src: Path) -> None:
    with pytest.raises(ValueError):
        _args(src=src)


@pytest.mark.parametrize("user", ["0:0", "root", "1000", "-1:0", ""])
def test_run_args_reject_root_or_malformed_user(user: str) -> None:
    with pytest.raises(ValueError):
        _args(user=user)


@pytest.mark.parametrize("image", ["", "--privileged"])
def test_run_args_reject_bad_image_names(image: str) -> None:
    with pytest.raises(ValueError):
        _args(image=image)


# --- run_capped: bounded output and timeouts (plain local processes) ----------


def test_run_capped_returns_output_and_exit_code() -> None:
    code, out, err, truncated = run_capped(
        [sys.executable, "-c", "import sys; print('hi'); print('oops', file=sys.stderr); sys.exit(3)"],
        timeout=10,
    )
    assert (code, out, err, truncated) == (3, b"hi\n", b"oops\n", False)


def test_run_capped_keeps_only_the_first_part_of_huge_output() -> None:
    code, out, _, truncated = run_capped(
        [sys.executable, "-c", f"import sys; sys.stdout.write('x' * {MAX_OUTPUT_BYTES * 3})"],
        timeout=30,
    )
    assert code == 0
    assert truncated
    assert len(out) == MAX_OUTPUT_BYTES


def test_run_capped_kills_commands_that_run_too_long() -> None:
    code, _, _, _ = run_capped([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert code is None


# --- exec(): argv must be a list ------------------------------------------------


@pytest.mark.parametrize("argv", ["ls -la", b"ls"])
def test_exec_rejects_a_single_string_as_argv(argv: str | bytes) -> None:
    # Unpacked into the docker command, "ls -la" would become 'l','s',' ','-',...
    # The check runs before any docker call, so this needs no Docker.
    with pytest.raises(TypeError, match="not a single string"):
        DockerLocal().exec("0" * 64, argv)  # type: ignore[arg-type]
