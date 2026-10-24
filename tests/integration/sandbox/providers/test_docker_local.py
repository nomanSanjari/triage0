"""DockerLocal against a real Docker daemon and the triage0-sandbox:dev image.

Build the image first:  docker build -t triage0-sandbox:dev - < Dockerfile.sandbox
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from triage0.sandbox.providers.docker_local import (
    MAX_OUTPUT_BYTES,
    DockerLocal,
    SandboxError,
    SandboxLimits,
    list_sandbox_containers,
)

IMAGE = "triage0-sandbox:dev"


def _image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True, timeout=30
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(not _image_available(), reason=f"docker or {IMAGE} not available")


@pytest.fixture
def provider() -> DockerLocal:
    return DockerLocal(SandboxLimits(memory="256m", lifetime_seconds=120, exec_timeout_seconds=30))


@pytest.fixture
def src(tmp_path: Path) -> Path:
    """A private (0700) folder, like the worktrees repo_cache creates."""
    folder = tmp_path / "wt"
    folder.mkdir(mode=0o700)
    (folder / "hello.c").write_text("int main(void) { return 0; }\n")
    return folder


@pytest.fixture
def container(provider: DockerLocal, src: Path) -> Iterator[str]:
    container_id = provider.start(IMAGE, src)
    try:
        yield container_id
    finally:
        provider.stop(container_id)
    assert container_id not in list_sandbox_containers()


def test_reads_the_mounted_private_worktree(provider: DockerLocal, container: str) -> None:
    result = provider.exec(container, ["cat", "/src/hello.c"])
    assert result.exit_code == 0
    assert "int main" in result.stdout


def test_has_no_network(provider: DockerLocal, container: str) -> None:
    result = provider.exec(container, ["ls", "/sys/class/net"])
    assert result.stdout.split() == ["lo"]


def test_runs_as_non_root(provider: DockerLocal, container: str) -> None:
    assert provider.exec(container, ["id", "-u"]).stdout.strip() != "0"


@pytest.mark.parametrize("path", ["/src/new.txt", "/src/hello.c", "/usr/evil"])
def test_cannot_write_outside_tmp(provider: DockerLocal, container: str, path: str) -> None:
    assert provider.exec(container, ["touch", path]).exit_code != 0


def test_can_write_to_tmp(provider: DockerLocal, container: str) -> None:
    assert provider.exec(container, ["touch", "/tmp/ok"]).exit_code == 0


def test_reports_the_commands_own_exit_code(provider: DockerLocal, container: str) -> None:
    result = provider.exec(container, ["ls", "/src/missing"])
    assert result.exit_code == 2
    assert "No such file" in result.stderr


def test_huge_output_is_truncated(provider: DockerLocal, container: str) -> None:
    result = provider.exec(container, ["head", "-c", str(MAX_OUTPUT_BYTES * 3), "/dev/zero"])
    assert result.truncated
    assert len(result.stdout) == MAX_OUTPUT_BYTES


def test_timeout_removes_the_container(provider: DockerLocal, container: str) -> None:
    result = provider.exec(container, ["sleep", "60"], timeout=2)
    assert result.timed_out and result.exit_code is None
    assert container not in list_sandbox_containers()
    with pytest.raises(SandboxError):
        provider.exec(container, ["true"])


def test_stop_is_safe_to_repeat(provider: DockerLocal, src: Path) -> None:
    container_id = provider.start(IMAGE, src)
    provider.stop(container_id)
    provider.stop(container_id)
    assert container_id not in list_sandbox_containers()
