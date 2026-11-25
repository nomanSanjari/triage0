"""Build-flag verdicts are made only from complete sealed sandbox output."""

from __future__ import annotations

import pytest

from triage0.checkers.build_flag_checker import BuildFlagChecker
from triage0.checkers.tools.build_tools import MAX_BUILD_FILE_BYTES
from triage0.core.evidence import Argv, CommandResult, DeterministicProof, EvidenceStatus
from triage0.core.interfaces import BaseClaimChecker
from triage0.core.models import BuildFlagClaim, BuildSystem, ClaimKind, SymbolClaim
from triage0.sandbox.providers.docker_local import SandboxError


class FakeSession:
    tree_hash = "a" * 40
    commit = "b" * 40

    def __init__(self, text: str, *, exit_code: int = 0, timed_out: bool = False, truncated: bool = False, fail: bool = False) -> None:
        self.text = text
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.truncated = truncated
        self.fail = fail
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: Argv, *, timeout_s: float | None = None) -> CommandResult:
        raise AssertionError("exploratory output must not decide the verdict")

    def seal(self, argv: Argv, *, timeout_s: float | None = None) -> DeterministicProof:
        self.calls.append(tuple(argv))
        assert timeout_s is not None
        if self.fail:
            raise SandboxError("not available")
        return DeterministicProof.from_command_result(
            CommandResult(tool="find", argv=argv, stdout=self.text, exit_code=self.exit_code, timed_out=self.timed_out, truncated=self.truncated),
            git_tree_hash=self.tree_hash,
        )


def claim(flag: str = "FEATURE", *, system: BuildSystem = BuildSystem.CMAKE, enabled: bool = True, **kwargs: str) -> BuildFlagClaim:
    return BuildFlagClaim(claim_id="1", statement="option default", flag=flag, build_system=system, expected_enabled=enabled, **kwargs)


def test_checker_interface_and_wrong_claim() -> None:
    checker = BuildFlagChecker()
    other = SymbolClaim(claim_id="2", statement="symbol", symbol="x")
    assert isinstance(checker, BaseClaimChecker)
    assert checker.claim_kind is ClaimKind.BUILD_FLAG
    assert checker.supports(claim()) and not checker.supports(other)
    with pytest.raises(TypeError):
        checker.check(other, FakeSession(""))


@pytest.mark.parametrize(("default", "expected", "status"), [
    ("ON", True, EvidenceStatus.CONFIRMED),
    ("OFF", False, EvidenceStatus.CONFIRMED),
    ("ON", False, EvidenceStatus.CONTRADICTED),
    ("OFF", True, EvidenceStatus.CONTRADICTED),
])
def test_cmake_literals_yield_a_sealed_verdict(default: str, expected: bool, status: EvidenceStatus) -> None:
    session = FakeSession(f'# header\noption(FEATURE "a feature" {default})\n')
    result = BuildFlagChecker().check(claim(enabled=expected), session)
    assert result.status is status
    assert result.proof is not None and result.proof.stdout == session.text
    assert result.proof.exit_code == 0
    assert "CMakeLists.txt:2" in (result.summary or "")
    assert result.proof.argv == ("find", "-P", "/src/CMakeLists.txt", "-maxdepth", "0", "-type", "f", "-exec", "cat", "--", "{}", "+")


@pytest.mark.parametrize(("macro", "flag", "given", "absent", "expected"), [
    ("ENABLE", "trace", "enableval", "yes", True),
    ("WITH", "tls", "withval", "no", False),
])
def test_autoconf_literal_defaults(macro: str, flag: str, given: str, absent: str, expected: bool) -> None:
    text = f'AC_INIT([project])\nAC_ARG_{macro}([{flag}], [help], [want=${given}], [want={absent}])\n'
    session = FakeSession(text)
    for assertion, status in [(expected, EvidenceStatus.CONFIRMED), (not expected, EvidenceStatus.CONTRADICTED)]:
        result = BuildFlagChecker().check(claim(flag, system=BuildSystem.AUTOCONF, enabled=assertion), session)
        assert result.status is status
        assert result.proof is not None and result.proof.stdout == text
        assert "configure.ac:2" in (result.summary or "")
        assert result.proof.argv[2] == "/src/configure.ac"


@pytest.mark.parametrize("text", [
    'if(WIN32)\noption(FEATURE "x" ON)\nendif()\n',
    'function(make_feature)\noption(FEATURE "x" ON)\nendfunction()\n',
    'option(FEATURE "x" ${SOME_DEFAULT})\n',
    'cmake_dependent_option(FEATURE "x" ON "WIN32" OFF)\n',
    'option(FEATURE "x" ON)\nif(WIN32)\nset(FEATURE OFF)\nendif()\n',
    'option(FEATURE "x" ON)\noption(FEATURE "x" OFF)\n',
    'option("FEATURE" "x" OFF)\noption(FEATURE "x" ON)\n',
    '# option(FEATURE "x" ON)\n',
    'message("option(FEATURE \\"x\\" ON)")\n',
    '#[[ option(FEATURE "x" ON) ]]\n',
    'set(OTHER [=[option(FEATURE "x" ON)]=])\n',
    'option(FEATURE "x" ON)\nif(BROKEN)\n',
    'option(FEATURE "x" ON)\nmessage("unclosed)\n',
    'option(FEATURE "x" ON)\noption(${OTHER} "x" OFF)\n',
    'option(FEATURE "x" ON)\n# invalid byte decoded as \ufffd\n',
    'option(FEATURE "x" ON)\nset("FEATURE" OFF)\n',
    'option(FEATURE "x" ON)\nset_property(CACHE FEATURE PROPERTY VALUE OFF)\n',
    'macro(option)\nendmacro()\noption(FEATURE "x" ON)\n',
    # An earlier included file or subdirectory can set the flag before
    # option() runs, so the literal root declaration is not the default.
    'include(cmake/defaults.cmake)\noption(FEATURE "x" ON)\n',
    'add_subdirectory(config)\noption(FEATURE "x" ON)\n',
    # The declaration can be unreachable.
    'return()\noption(FEATURE "x" ON)\n',
    'message(FATAL_ERROR "stop")\noption(FEATURE "x" ON)\n',
    # Nested control flow is still conditional.
    'if(WIN32)\nif(MSVC)\noption(FEATURE "x" ON)\nendif()\nendif()\n',
    # A dynamic invocation that never spells the option command.
    'cmake_language(CALL ${COMMAND} ${NAME} ON)\noption(FEATURE "x" ON)\n',
])
def test_cmake_ambiguous_or_missing_is_never_a_terminal_verdict(text: str) -> None:
    result = BuildFlagChecker().check(claim(), FakeSession(text))
    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None


@pytest.mark.parametrize("text", [
    'if test "$host" = x; then\nAC_ARG_WITH(tls, [], [x=$withval], [x=no])\nfi\n',
    'case "$host" in\n  linux*)\nAC_ARG_WITH(tls, [], [x=$withval], [x=no])\n  ;;\nesac\n',
    'if true; then\necho fi\nAC_ARG_WITH(tls, [], [x=$withval], [x=no])\nfi\n',
    'm4_if([x], [y], [\nAC_ARG_WITH(tls, [], [x=$withval], [x=no])\n])\n',
    '[\nAC_ARG_WITH(tls, [], [x=$withval], [x=no])\n]\n',
    'AC_DEFUN([macro], [\nAC_ARG_WITH(tls, [], [x=$withval], [x=no])\n])\n',
    'AC_ARG_WITH(tls, [], [x=$withval], [x=$platform])\n',
    'AC_ARG_WITH(tls, [], [x=$withval], [x = no])\n',
    'AC_ARG_WITH(tls, [], [x=$withval], [if test x; then x=yes; else x=no; fi])\n',
    'AC_ARG_WITH(tls, [], [x=$withval])\n',
    'AC_ARG_WITH(tls, [], [x=$withval], [x=no])\nif test x; then\n  x=yes\nfi\n',
    'AC_ARG_WITH(tls, [], [x=$withval], [x=no])\nAC_ARG_ENABLE(tls, [], [y=$enableval], [y=yes])\n',
    # Indirect writes to the selected variable are not literal defaults.
    "AC_ARG_ENABLE(tls, [], [x=$enableval], [x=no])\neval \"x\"'=yes'\n",
    'AC_ARG_ENABLE(tls, [], [x=$enableval], [x=no])\nbuiltin eval "x"\'=yes\'\n',
    'AC_ARG_ENABLE(tls, [], [x=$enableval], [x=no])\nsource ./platform-defaults\n',
    'AC_ARG_ENABLE(tls, [], [x=$enableval], [x=no])\n. ./platform-defaults\n',
    'AC_ARG_ENABLE(tls, [], [x=$enableval], [x=no])\nread x < platform-defaults\n',
    'AC_ARG_ENABLE(tls, [], [x=$enableval], [x=no])\nexport x\n',
    'AC_ARG_ENABLE(tls, [], [x=$enableval], [x=no])\nprintf -v x yes\n',
    'apply() {\n  x=yes\n}\nAC_ARG_ENABLE(tls, [], [x=$enableval], [x=no])\napply\n',
])
def test_autoconf_uncertainty_is_not_a_verdict(text: str) -> None:
    result = BuildFlagChecker().check(claim("tls", system=BuildSystem.AUTOCONF), FakeSession(text))
    assert result.status is EvidenceStatus.CANT_CHECK


@pytest.mark.parametrize("flag", ['--help; touch /tmp/evil', '$(touch /tmp/evil)', '-FEATURE', 'FEATURE\noption(X "" ON)'])
def test_untrusted_flags_never_reach_the_command(flag: str) -> None:
    benign, hostile = FakeSession('option(FEATURE "x" ON)\n'), FakeSession('option(FEATURE "x" ON)\n')
    BuildFlagChecker().check(claim(), benign)
    result = BuildFlagChecker().check(claim(flag), hostile)
    assert result.status is EvidenceStatus.CANT_CHECK
    assert benign.calls == hostile.calls


@pytest.mark.parametrize("options", [dict(exit_code=1), dict(timed_out=True), dict(truncated=True), dict(fail=True)])
def test_failed_or_incomplete_sealed_read_is_not_a_verdict(options: dict[str, object]) -> None:
    session = FakeSession('option(FEATURE "x" ON)\n', **options)  # type: ignore[arg-type]
    assert BuildFlagChecker().check(claim(), session).status is EvidenceStatus.CANT_CHECK


def test_missing_build_file_and_unsupported_scope_abstain() -> None:
    session = FakeSession("")
    assert BuildFlagChecker().check(claim(), session).status is EvidenceStatus.CANT_CHECK
    for c in (claim(system=BuildSystem.MESON), claim(config="Debug"), claim(target="test")):
        session.calls.clear()
        assert BuildFlagChecker().check(c, session).status is EvidenceStatus.CANT_CHECK
        assert not session.calls


def test_an_include_after_the_declaration_still_confirms() -> None:
    """Only an earlier include can change what option() does."""
    text = 'option(FEATURE "x" ON)\ninclude(cmake/later.cmake)\n'

    result = BuildFlagChecker().check(claim(), FakeSession(text))

    assert result.status is EvidenceStatus.CONFIRMED


def test_an_oversized_build_file_abstains_before_parsing() -> None:
    text = 'option(FEATURE "x" ON)\n' + "#" * MAX_BUILD_FILE_BYTES

    result = BuildFlagChecker().check(claim(), FakeSession(text))

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.proof is not None and result.proof.exit_code == 0
