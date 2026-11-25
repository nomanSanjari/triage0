"""Check curl 8.13.0's block_ip example: main calls free_connection_filter
at line 348, which calls ip_list_free_all at line 188. The example compiles
with the checked-in headers, without running the project's build scripts.
"""

from __future__ import annotations

import os

import pytest

from triage0.checkers.call_path_checker import CallPathChecker
from triage0.core.evidence import EvidenceResult, EvidenceStatus
from triage0.core.models import CallPathClaim
from triage0.sandbox.manager import Session

from ..sandbox.providers.test_docker_local import IMAGE, _image_available

pytestmark = [
    pytest.mark.skipif(not _image_available(), reason=f"docker or {IMAGE} not available"),
    pytest.mark.skipif(
        os.environ.get("TRIAGE0_NETWORK_TESTS") != "1",
        reason="set TRIAGE0_NETWORK_TESTS=1 (clones curl)",
    ),
]


def _from_main(session: Session, target: str) -> EvidenceResult:
    return CallPathChecker().check(
        CallPathClaim(
            claim_id="curl",
            statement="main reaches the target",
            file_hint="docs/examples/block_ip.c",
            entrypoint="main",
            target=target,
        ),
        session,
    )


def test_curl_direct_path(session_curl_8_13_0: Session) -> None:
    result = _from_main(session_curl_8_13_0, "ip_list_free_all")

    assert result.proof is not None
    assert result.status is EvidenceStatus.CONFIRMED, (result.summary, result.proof.stderr)
    assert '["main","free_connection_filter",348]' in result.proof.stdout
    assert '["free_connection_filter","ip_list_free_all",188]' in result.proof.stdout
    assert "main -> free_connection_filter (docs/examples/block_ip.c:348)" in (result.summary or "")
    assert "free_connection_filter -> ip_list_free_all (docs/examples/block_ip.c:188)" in (result.summary or "")


def test_a_real_function_outside_the_reachable_set_is_not_confirmed(
    session_curl_8_13_0: Session,
) -> None:
    """ip_match is called from opensocket only, which main never calls directly."""

    result = _from_main(session_curl_8_13_0, "ip_match")

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    # Clang parsed the file, so this is "no witnessed path", not a failed run.
    assert result.proof is not None and result.proof.exit_code == 0


def test_a_made_up_target_is_not_confirmed(session_curl_8_13_0: Session) -> None:
    result = _from_main(session_curl_8_13_0, "curl_no_such_function")

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
