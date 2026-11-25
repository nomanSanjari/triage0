"""Syntax confirmation uses actual tree-sitter, not just the ctags index."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from triage0.checkers.symbol_checker import SymbolChecker
from triage0.core.evidence import EvidenceStatus
from triage0.core.models import SymbolClaim, SymbolKind
from triage0.sandbox.manager import Session
from triage0.sandbox.providers.docker_local import DockerLocal, SandboxLimits

from ..sandbox.providers.test_docker_local import IMAGE, _image_available

pytestmark = pytest.mark.skipif(not _image_available(), reason=f"docker or {IMAGE} not available")


@pytest.fixture
def source_session(tmp_path: Path) -> Iterator[Session]:
    src = tmp_path / "src"
    src.mkdir()
    (src / "test.c").write_text(
        "int valid(void) { return 1; }\n"
        "int declared(void);\n"
        "struct record { int member; };\n"
        "typedef int number;\n"
        "static int count = 1;\n"
        "int broken(void) { return 2 }\n"
    )
    (src / "test.h").write_text("#ifndef FLAG\n#define FLAG 1\n#endif\n")
    provider = DockerLocal(SandboxLimits(memory="256m", lifetime_seconds=120))
    container_id: str | None = None
    try:
        container_id = provider.start(IMAGE, src)
        yield Session("test", "a" * 40, "b" * 40, container_id, provider)
    finally:
        if container_id is not None:
            provider.stop(container_id)


def _claim(symbol: str, kind: SymbolKind, *, present: bool = True) -> SymbolClaim:
    return SymbolClaim(
        claim_id="syntax", statement="check the source", symbol=symbol,
        symbol_kind=kind, expected_present=present,
    )


def test_syntax_errors_are_local_and_do_not_validate_broken_definitions(
    source_session: Session,
) -> None:
    checker = SymbolChecker()
    valid = checker.check(_claim("valid", SymbolKind.FUNCTION), source_session)
    broken = checker.check(_claim("broken", SymbolKind.FUNCTION, present=False), source_session)
    declaration = checker.check(_claim("declared", SymbolKind.FUNCTION), source_session)
    assert valid.status is EvidenceStatus.CONFIRMED
    assert valid.proof is not None and valid.proof.tool == "tree-sitter"
    assert broken.status is EvidenceStatus.CANT_CHECK
    assert broken.missing is None
    assert broken.proof is not None and broken.proof.tool == "tree-sitter"
    assert declaration.status is EvidenceStatus.CANT_CHECK


def test_guarded_macro_is_a_source_definition_not_a_build_assertion(source_session: Session) -> None:
    result = SymbolChecker().check(_claim("FLAG", SymbolKind.MACRO, present=False), source_session)
    assert result.status is EvidenceStatus.CONTRADICTED
    assert result.proof is not None and result.proof.tool == "tree-sitter"
    assert "in source" in (result.summary or "")


@pytest.mark.parametrize(
    ("name", "kind"),
    [("record", SymbolKind.STRUCT), ("number", SymbolKind.TYPE), ("count", SymbolKind.VARIABLE)],
)
def test_other_indexed_kinds_are_checked_against_their_syntax(
    source_session: Session, name: str, kind: SymbolKind,
) -> None:
    result = SymbolChecker().check(_claim(name, kind), source_session)
    assert result.status is EvidenceStatus.CONFIRMED
    assert result.proof is not None and result.proof.tool == "tree-sitter"
