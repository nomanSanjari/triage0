"""The literal declarations below were inspected in curl-8_13_0 first.

CMakeLists.txt:212: option(BUILD_CURL_EXE "Build curl executable" ON), but
CMakeLists.txt:70-73 include other files before that line, so the CMake case
must abstain rather than report the root-file spelling as the default.
configure.ac:2330-2333: AC_ARG_WITH(libssh2, ..., OPT_LIBSSH2=$withval,
                                  OPT_LIBSSH2=no)
configure.ac:685-713: http uses a case action and AC_MSG_RESULT, not a
literal assignment in its absent branch; this checker intentionally abstains.
"""

from __future__ import annotations

import os

import pytest

from triage0.checkers.build_flag_checker import BuildFlagChecker
from triage0.core.evidence import EvidenceStatus
from triage0.core.models import BuildFlagClaim, BuildSystem
from triage0.sandbox.manager import Session

from ..sandbox.providers.test_docker_local import IMAGE, _image_available

pytestmark = [
    pytest.mark.skipif(not _image_available(), reason=f"docker or {IMAGE} not available"),
    pytest.mark.skipif(os.environ.get("TRIAGE0_NETWORK_TESTS") != "1", reason="set TRIAGE0_NETWORK_TESTS=1 (clones curl)"),
]


@pytest.mark.parametrize(("system", "flag", "enabled", "line", "literal"), [
    (BuildSystem.AUTOCONF, "--with-libssh2", False, "configure.ac:2330", "OPT_LIBSSH2=$withval, OPT_LIBSSH2=no)"),
])
def test_real_curl_literal_defaults(
    session_curl_8_13_0: Session, system: BuildSystem, flag: str, enabled: bool, line: str, literal: str,
) -> None:
    checker = BuildFlagChecker()
    for expected, status in [(enabled, EvidenceStatus.CONFIRMED), (not enabled, EvidenceStatus.CONTRADICTED)]:
        result = checker.check(BuildFlagClaim(claim_id="curl", statement="default", build_system=system, flag=flag, expected_enabled=expected), session_curl_8_13_0)
        assert result.status is status
        assert result.proof is not None and result.proof.exit_code == 0
        assert not result.proof.timed_out and not result.proof.truncated
        assert literal in result.proof.stdout
        assert line in (result.summary or "")


def test_real_curl_cmake_abstains_when_files_are_included_first(
    session_curl_8_13_0: Session,
) -> None:
    """CMakeLists.txt includes Utilities and Macros before option() runs."""
    result = BuildFlagChecker().check(
        BuildFlagClaim(
            claim_id="curl",
            statement="default",
            build_system=BuildSystem.CMAKE,
            flag="BUILD_CURL_EXE",
        ),
        session_curl_8_13_0,
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert result.proof is not None and result.proof.exit_code == 0


def test_real_http_does_not_infer_a_default_from_a_message(session_curl_8_13_0: Session) -> None:
    result = BuildFlagChecker().check(
        BuildFlagClaim(claim_id="curl", statement="http", build_system=BuildSystem.AUTOCONF, flag="--enable-http"),
        session_curl_8_13_0,
    )
    assert result.status is EvidenceStatus.CANT_CHECK
