"""Direct call paths in a local checkout: real Clang, Docker, no network.

The curl test proves the checker on an upstream project, but it is skipped
unless TRIAGE0_NETWORK_TESTS=1. These fixtures keep the projection itself
under test in every run that has the image, including the cases where the
checker must abstain.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from triage0.checkers.call_path_checker import CallPathChecker
from triage0.core.evidence import EvidenceResult, EvidenceStatus
from triage0.core.models import CallPathClaim
from triage0.sandbox.manager import Session
from triage0.sandbox.providers.docker_local import DockerLocal, SandboxLimits

from ..sandbox.providers.test_docker_local import IMAGE, _image_available

pytestmark = pytest.mark.skipif(not _image_available(), reason=f"docker or {IMAGE} not available")

# Each file is its own translation unit. Which names exist and where they are
# called was read from the fixtures below, not assumed.
FILES = {
    "direct.c": "int target(void){return 1;}\nint entry(void){return target();}\n",
    "twohop.c": (
        "int target(void){return 1;}\n"
        "int middle(void){return target();}\n"
        "int entry(void){return middle();}\n"
    ),
    "recursive.c": "int entry(int n){return n ? entry(n - 1) : 0;}\n",
    # The call refers to the prototype; the definition comes later.
    "prototype.c": (
        "int target(void);\n"
        "int entry(void){return target();}\n"
        "int target(void){return 1;}\n"
    ),
    "indirect.c": (
        "int target(void){return 1;}\n"
        "int entry(void){int (*call)(void) = target; return call();}\n"
    ),
    # The call site is inside the macro expansion, so it has no bare location.
    "macro.c": (
        "#define CALLIT() target()\n"
        "int target(void){return 1;}\n"
        "int entry(void){ CALLIT(); return 0; }\n"
    ),
    "header.h": "static int from_header(void){return 1;}\n",
    "header.c": '#include "header.h"\nint entry(void){return from_header();}\n',
    # A #line directive changes the presumed numbering, not the physical one.
    "line.c": (
        '#line 200 "other.c"\n'
        "int target(void){return 1;}\n"
        "int entry(void){return target();}\n"
    ),
}


def check(
    session: Session, path: str, entrypoint: str, target: str
) -> EvidenceResult:
    return CallPathChecker().check(
        CallPathClaim(
            claim_id="local",
            statement="the entry point reaches the target",
            file_hint=path,
            entrypoint=entrypoint,
            target=target,
        ),
        session,
    )


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Session]:
    src = tmp_path_factory.mktemp("src")
    for name, text in FILES.items():
        (src / name).write_text(text)
    provider = DockerLocal(
        SandboxLimits(memory="256m", lifetime_seconds=300, exec_timeout_seconds=30)
    )
    container_id: str | None = None
    try:
        container_id = provider.start(IMAGE, src)
        yield Session("local", "a" * 40, "b" * 40, container_id, provider)
    finally:
        if container_id is not None:
            provider.stop(container_id)


def test_a_direct_call_is_confirmed(session: Session) -> None:
    result = check(session, "direct.c", "entry", "target")

    assert result.status is EvidenceStatus.CONFIRMED
    assert result.proof is not None and result.proof.exit_code == 0
    assert "entry -> target (direct.c:2)" in (result.summary or "")


def test_a_two_hop_path_cites_every_hop(session: Session) -> None:
    result = check(session, "twohop.c", "entry", "target")

    assert result.status is EvidenceStatus.CONFIRMED
    assert "entry -> middle (twohop.c:3)" in (result.summary or "")
    assert "middle -> target (twohop.c:2)" in (result.summary or "")


def test_a_later_definition_resolves_the_prototype_reference(session: Session) -> None:
    result = check(session, "prototype.c", "entry", "target")

    assert result.status is EvidenceStatus.CONFIRMED
    assert "entry -> target (prototype.c:2)" in (result.summary or "")


def test_a_recursive_call_is_its_own_hop(session: Session) -> None:
    result = check(session, "recursive.c", "entry", "entry")

    assert result.status is EvidenceStatus.CONFIRMED
    assert "entry -> entry (recursive.c:1)" in (result.summary or "")


def test_physical_lines_win_over_a_line_directive(session: Session) -> None:
    result = check(session, "line.c", "entry", "target")

    assert result.status is EvidenceStatus.CONFIRMED
    assert "line.c:3" in (result.summary or "")
    # 201 would be the presumed line after "#line 200".
    assert "201" not in (result.summary or "")


@pytest.mark.parametrize(
    ("path", "entrypoint", "target"),
    [
        # A call through a function pointer is not a direct call.
        ("indirect.c", "entry", "target"),
        # The call site sits inside a macro expansion.
        ("macro.c", "entry", "target"),
        # The definition lives in a header, so this unit defines no body.
        ("header.c", "entry", "from_header"),
        # A name that no function in the unit has.
        ("direct.c", "entry", "missing"),
    ],
)
def test_what_the_projection_cannot_see_is_not_confirmed(
    session: Session, path: str, entrypoint: str, target: str
) -> None:
    """Clang must have parsed the file; the verdict is still CANT_CHECK."""

    result = check(session, path, entrypoint, target)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert result.proof is not None and result.proof.exit_code == 0
