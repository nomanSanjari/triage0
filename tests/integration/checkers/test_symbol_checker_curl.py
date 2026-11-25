"""SymbolChecker against real curl, in the real image, with networking off.

Needs Docker and the image:  docker build -t triage0-sandbox:dev - < Dockerfile.sandbox
Also needs internet (it clones curl):  TRIAGE0_NETWORK_TESTS=1

The unit tests use a fake sandbox and hand-written tags. These tests are the
ones that prove the checker survives a real 12k-tag index: the wide ctags run
truncates at the sandbox cap, so this exercises the chunked scan and the
truncation flag that no fake can be trusted to reproduce.
"""

from __future__ import annotations

import os

import pytest

from triage0.checkers.symbol_checker import SymbolChecker
from triage0.core.evidence import EvidenceStatus, MissingEvidence
from triage0.core.models import SymbolClaim, SymbolKind
from triage0.sandbox.manager import SandboxManager, Session

from ..sandbox.providers.test_docker_local import IMAGE, _image_available
from .conftest import URL

pytestmark = [
    pytest.mark.skipif(not _image_available(), reason=f"docker or {IMAGE} not available"),
    pytest.mark.skipif(
        os.environ.get("TRIAGE0_NETWORK_TESTS") != "1",
        reason="set TRIAGE0_NETWORK_TESTS=1 (clones curl)",
    ),
]


def _claim(
    symbol: str,
    kind: SymbolKind,
    *,
    file_hint: str | None = None,
    expected_present: bool = True,
) -> SymbolClaim:
    return SymbolClaim(
        claim_id="curl",
        statement=f"{symbol} should exist",
        symbol=symbol,
        symbol_kind=kind,
        file_hint=file_hint,
        expected_present=expected_present,
    )


def test_a_function_in_a_named_file_is_confirmed(session_curl_8_13_0: Session) -> None:
    """Issue 2.3b: a real function is confirmed, and the proof is sealed."""
    result = SymbolChecker().check(
        _claim("Curl_dyn_add", SymbolKind.FUNCTION, file_hint="lib/dynbuf.c"),
        session_curl_8_13_0,
    )

    assert result.status is EvidenceStatus.CONFIRMED
    assert result.proof is not None
    assert result.proof.tool == "tree-sitter"
    assert result.proof.exit_code == 0
    assert not result.proof.truncated
    assert not result.proof.timed_out
    assert "Curl_dyn_add" in (result.proof.stdout or "")
    assert "lib/dynbuf.c:180" in (result.summary or "")


def test_a_macro_in_a_header_is_confirmed(session_curl_8_13_0: Session) -> None:
    """An #ifndef-guarded definition exists in source despite errors elsewhere."""
    result = SymbolChecker().check(
        _claim("CURL_MAX_WRITE_SIZE", SymbolKind.MACRO, file_hint="include/curl/curl.h"),
        session_curl_8_13_0,
    )

    assert result.status is EvidenceStatus.CONFIRMED
    assert result.proof is not None
    assert result.proof.tool == "tree-sitter"
    assert result.proof.exit_code == 0
    assert not result.proof.truncated
    assert "CURL_MAX_WRITE_SIZE" in (result.proof.stdout or "")
    assert "include/curl/curl.h:266" in (result.summary or "")


def test_a_symbol_that_never_existed_is_absent_not_contradicted(
    session_curl_8_13_0: Session,
) -> None:
    """A full-tree scan must finish before absence can be claimed at all."""
    result = SymbolChecker().check(
        _claim("Curl_cookie_parse_fast", SymbolKind.FUNCTION),
        session_curl_8_13_0,
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is MissingEvidence.SYMBOL_ABSENT
    # The whole tree was read: the wide index is 12k tags, far past the cap.
    assert "indexed" in (result.summary or "")


def test_a_symbol_the_claim_calls_absent_is_contradicted(session_curl_8_13_0: Session) -> None:
    result = SymbolChecker().check(
        _claim("Curl_dyn_add", SymbolKind.UNKNOWN, expected_present=False),
        session_curl_8_13_0,
    )

    assert result.status is EvidenceStatus.CONTRADICTED
    assert result.proof is not None


def test_a_renamed_symbol_is_absent_at_the_newer_tag(manager: SandboxManager) -> None:
    """curl 8.14.0 renamed Curl_dyn_add to curlx_dyn_add.

    The old name reports "absent", never "contradicted": a rename is exactly
    the case a history follow-up exists for.
    """
    with manager.open(URL, "curl-8_14_0") as session:
        old = SymbolChecker().check(_claim("Curl_dyn_add", SymbolKind.FUNCTION), session)
        new = SymbolChecker().check(_claim("curlx_dyn_add", SymbolKind.FUNCTION), session)

    assert old.status is EvidenceStatus.CANT_CHECK
    assert old.missing is MissingEvidence.SYMBOL_ABSENT
    # The other half of the rename: the new name is there, deterministically.
    assert new.status is EvidenceStatus.CONFIRMED
    assert new.proof is not None and new.proof.exit_code == 0
