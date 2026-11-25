"""File-line safety checks against real sandbox filesystem behavior."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from triage0.checkers.file_line_checker import FileLineChecker
from triage0.core.evidence import EvidenceResult, EvidenceStatus
from triage0.core.models import FileLineClaim
from triage0.sandbox.manager import Session
from triage0.sandbox.providers.docker_local import DockerLocal, SandboxLimits

from ..sandbox.providers.test_docker_local import IMAGE, _image_available

pytestmark = pytest.mark.skipif(not _image_available(), reason=f"docker or {IMAGE} not available")


@pytest.fixture
def session_with_files(tmp_path: Path) -> Iterator[Session]:
    src = tmp_path / "src"
    src.mkdir()
    (src / "escape").symlink_to("/etc/passwd")
    (src / "invalid.c").write_bytes(b"hello \xff\n")
    private = src / "private"
    private.mkdir()
    (private / "x.c").write_text("int x;\n")
    private.chmod(0)
    provider = DockerLocal(SandboxLimits(memory="256m", lifetime_seconds=120, exec_timeout_seconds=30))
    container_id: str | None = None
    try:
        container_id = provider.start(IMAGE, src)
        yield Session("test", "a" * 40, "b" * 40, container_id, provider)
    finally:
        private.chmod(0o700)
        if container_id is not None:
            provider.stop(container_id)


def check(session: Session, path: str, snippet: str | None = None) -> EvidenceResult:
    claim = FileLineClaim(
        claim_id="c", statement="check source", path=path, line_start=1,
        expected_snippet=snippet,
    )
    return FileLineChecker().check(claim, session)


def test_symlink_outside_checkout_is_not_read(session_with_files: Session) -> None:
    result = check(session_with_files, "escape", "root:")
    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert result.proof is None


def test_inaccessible_parent_is_not_file_absent(session_with_files: Session) -> None:
    result = check(session_with_files, "private/x.c")
    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None


def test_invalid_utf8_cannot_manufacture_a_match(session_with_files: Session) -> None:
    result = check(session_with_files, "invalid.c", "hello \ufffd")
    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
