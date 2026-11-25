"""Milestone 2 tools inside the real image, with networking off.

The checkers can only use tools that are already in the image: the container has
no network, so nothing can be installed at check time. These tests fail on the
pre-2.3a image, where tree-sitter-cli could not parse C at all.

Build the image first:  docker build -t triage0-sandbox:dev - < Dockerfile.sandbox
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from triage0.sandbox.providers.docker_local import DockerLocal, SandboxLimits

from .providers.test_docker_local import IMAGE, _image_available

pytestmark = pytest.mark.skipif(not _image_available(), reason=f"docker or {IMAGE} not available")

CONFIG = "/opt/triage0/config.json"

# A macro, a function definition and a call: real C is rarely simpler. These are
# the shapes 2.3d walks, and what 2.3b can use to confirm a ctags kind.
# It lives in the mounted worktree because the root filesystem is read-only
# and DockerLocal.exec() has no stdin.
SAMPLE_C = """\
#define SAMPLE_LIMIT 8
static int add(int a, int b) { return a + b; }
int caller(int y) { return add(y, y); }
"""


@pytest.fixture
def provider() -> DockerLocal:
    return DockerLocal(SandboxLimits(memory="256m", lifetime_seconds=120, exec_timeout_seconds=30))


@pytest.fixture
def src(tmp_path: Path) -> Path:
    folder = tmp_path / "wt"
    folder.mkdir(mode=0o700)
    (folder / "sample.c").write_text(SAMPLE_C)
    return folder


@pytest.fixture
def container(provider: DockerLocal, src: Path) -> Iterator[str]:
    container_id = provider.start(IMAGE, src)
    try:
        yield container_id
    finally:
        provider.stop(container_id)


def test_tree_sitter_parses_c_with_no_network(provider: DockerLocal, container: str) -> None:
    result = provider.exec(
        container, ["tree-sitter", "parse", "--config-path", CONFIG, "/src/sample.c"]
    )
    assert result.exit_code == 0, result.stderr
    # The parse tree is the evidence that the grammar was found and loaded.
    assert "preproc_def" in result.stdout
    assert "function_definition" in result.stdout
    assert "call_expression" in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["ctags", "--version"],
        ["clang", "--version"],
        ["tree-sitter", "--version"],
        ["git", "--version"],
        ["rg", "--version"],
    ],
)
def test_every_checker_tool_prints_a_version(
    provider: DockerLocal, container: str, argv: list[str]
) -> None:
    result = provider.exec(container, argv)
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip()
