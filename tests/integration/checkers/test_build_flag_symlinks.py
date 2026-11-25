"""Build files are read as regular files: symlinks are refused, plain files work."""

from __future__ import annotations

from pathlib import Path

import pytest

from triage0.checkers.build_flag_checker import BuildFlagChecker
from triage0.core.evidence import EvidenceStatus
from triage0.core.models import BuildFlagClaim, BuildSystem
from triage0.sandbox.manager import Session
from triage0.sandbox.providers.docker_local import DockerLocal

from ..sandbox.providers.test_docker_local import IMAGE, _image_available

pytestmark = pytest.mark.skipif(not _image_available(), reason=f"docker or {IMAGE} not available")


@pytest.mark.parametrize(("system", "name", "target", "source"), [
    (BuildSystem.CMAKE, "CMakeLists.txt", "other.cmake", 'option(FEATURE "x" ON)\n'),
    (BuildSystem.CMAKE, "CMakeLists.txt", "/etc/passwd", ""),
    (BuildSystem.AUTOCONF, "configure.ac", "other.ac", 'AC_ARG_WITH(tls, [], [x=$withval], [x=no])\n'),
    (BuildSystem.AUTOCONF, "configure.ac", "/etc/passwd", ""),
])
def test_build_file_symlinks_are_not_followed(tmp_path: Path, system: BuildSystem, name: str, target: str, source: str) -> None:
    if source:
        (tmp_path / target).write_text(source)
    (tmp_path / name).symlink_to(target)
    provider = DockerLocal()
    container_id = provider.start(IMAGE, tmp_path)
    try:
        session = Session(tag="test", tree_hash="a" * 40, commit="b" * 40, container_id=container_id, provider=provider)
        result = BuildFlagChecker().check(
            BuildFlagClaim(claim_id="1", statement="default", build_system=system, flag="FEATURE" if system is BuildSystem.CMAKE else "tls"),
            session,
        )
        assert result.status is EvidenceStatus.CANT_CHECK
        assert result.proof is not None and result.proof.stdout == ""
    finally:
        provider.stop(container_id)


def test_a_regular_build_file_is_still_read_and_confirmed(tmp_path: Path) -> None:
    """The symlink guard must not turn every normal checkout into a refusal."""
    (tmp_path / "CMakeLists.txt").write_text('option(FEATURE "x" ON)\n')
    (tmp_path / "configure.ac").write_text("AC_ARG_ENABLE(tls, [], [x=$enableval], [x=no])\n")
    provider = DockerLocal()
    container_id = provider.start(IMAGE, tmp_path)
    try:
        session = Session(tag="test", tree_hash="a" * 40, commit="b" * 40, container_id=container_id, provider=provider)
        for system, flag, expected, declaration in [
            (BuildSystem.CMAKE, "FEATURE", True, 'option(FEATURE "x" ON)'),
            (BuildSystem.AUTOCONF, "tls", False, "AC_ARG_ENABLE(tls, [], [x=$enableval], [x=no])"),
        ]:
            result = BuildFlagChecker().check(
                BuildFlagClaim(
                    claim_id="1",
                    statement="default",
                    build_system=system,
                    flag=flag,
                    expected_enabled=expected,
                ),
                session,
            )
            assert result.status is EvidenceStatus.CONFIRMED
            assert result.proof is not None and declaration in result.proof.stdout
    finally:
        provider.stop(container_id)
