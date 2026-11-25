"""Direct tests for the CMake/Autoconf declaration readers.

These helpers hold nearly all of the parsing complexity, so they are tested
here even though the checker is their only caller. Each case either proves an
accepted literal declaration or proves that an unsupported construct abstains.
"""

from __future__ import annotations

import pytest

from triage0.checkers.tools.build_tools import (
    MAX_BUILD_FILE_BYTES,
    Default,
    _autoconf_default,
    _cmake_commands,
    _cmake_default,
    _m4_call,
    _unconditional_shell,
    _unquote_m4,
    parse_default,
)
from triage0.core.models import BuildSystem

CMAKE_BASE = 'option(FEATURE "A feature" ON)\n'


# --- CMake lexical split -------------------------------------------------


def test_cmake_comments_and_strings_do_not_create_commands() -> None:
    commands = _cmake_commands('# option(FEATURE "x" ON)\nmessage(STATUS "option(FEATURE x ON)")\n')

    assert commands == [("message", 'STATUS "option(FEATURE x ON)"', 2)]


def test_cmake_bracket_literal_is_one_argument_not_a_command() -> None:
    commands = _cmake_commands('set(X [=[ option(FEATURE "x" ON) ]=])\n')

    assert commands is not None
    assert [name for name, _, _ in commands] == ["set"]


@pytest.mark.parametrize(
    "text",
    [
        'option(FEATURE "x" ON',  # unclosed command
        'option(FEATURE "x" ON)\nmessage("unclosed)\n',  # unclosed string
        "option(FEATURE \"x\" ON)\nset(X [=[ unclosed\n",  # unclosed bracket
        "not-a-command\n",
    ],
)
def test_cmake_malformed_lexing_returns_none(text: str) -> None:
    assert _cmake_commands(text) is None


# --- CMake declaration defaults -----------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (CMAKE_BASE, Default(True, 1)),
        ('option("FEATURE" "A feature" OFF)\n', Default(False, 1)),
        (f"message(STATUS start)\n{CMAKE_BASE}", Default(True, 2)),
        # include() after the declaration cannot change the default anymore.
        (CMAKE_BASE + "include(cmake/later.cmake)\n", Default(True, 1)),
        # return() after the declaration has already had no effect.
        (CMAKE_BASE + "return()\n", Default(True, 1)),
        # return() inside a function body does not stop the root file.
        (CMAKE_BASE + "function(f)\nreturn()\nendfunction()\n", Default(True, 1)),
        # SEND_ERROR continues, so the declaration is still reached.
        (f"message(SEND_ERROR \"x\")\n{CMAKE_BASE}", Default(True, 2)),
    ],
)
def test_cmake_literal_declarations_are_accepted(text: str, expected: Default) -> None:
    assert _cmake_default(text, "FEATURE") == expected


@pytest.mark.parametrize(
    "text",
    [
        # Earlier files can set the flag before option() runs (CMP0077).
        'include(cmake/defaults.cmake)\noption(FEATURE "A feature" ON)\n',
        'add_subdirectory(config)\noption(FEATURE "A feature" ON)\n',
        # The declaration is never reached.
        'return()\noption(FEATURE "A feature" ON)\n',
        'message(FATAL_ERROR "stop")\noption(FEATURE "A feature" ON)\n',
        # Dynamic invocation, including a constructed option name.
        "set(A FE)\nset(B ATURE)\ncmake_language(CALL option ${A}${B} ON)\n"
        'option(FEATURE "A feature" ON)\n',
        # A conditional or redefined declaration is not a literal default.
        'if(WIN32)\nif(MSVC)\noption(FEATURE "A feature" ON)\nendif()\nendif()\n',
        'function(make_feature)\noption(FEATURE "A feature" ON)\nendfunction()\n',
        'macro(option)\nendmacro()\noption(FEATURE "A feature" ON)\n',
        'set(FEATURE OFF)\noption(FEATURE "A feature" ON)\n',
        'option(FEATURE "A feature" ON)\noption(FEATURE "A feature" OFF)\n',
        'cmake_dependent_option(FEATURE "A feature" ON "WIN32" OFF)\n',
    ],
)
def test_cmake_uncertain_declarations_abstain(text: str) -> None:
    assert _cmake_default(text, "FEATURE") is None


# --- m4 helpers ----------------------------------------------------------


def test_m4_call_splits_at_top_level_commas_only() -> None:
    result = _m4_call("FOO(a, [b, c], (d, e))", 3)

    assert result is not None
    args, end = result
    assert [arg.strip() for arg in args] == ["a", "[b, c]", "(d, e)"]
    assert end == len("FOO(a, [b, c], (d, e))")


def test_m4_call_rejects_an_unclosed_call() -> None:
    assert _m4_call("FOO(a, [b, c]", 3) is None


@pytest.mark.parametrize(("value", "expected"), [("[ value ]", "value"), ("plain", "plain")])
def test_m4_unquote_removes_one_level_of_brackets(value: str, expected: str) -> None:
    assert _unquote_m4(value) == expected


# --- Autoconf declarations ----------------------------------------------


@pytest.mark.parametrize(
    ("text", "flag", "expected"),
    [
        (
            "AC_ARG_ENABLE(tls, [help], [want=$enableval], [want=no])\n",
            "--enable-tls",
            Default(False, 1),
        ),
        (
            "AC_ARG_WITH(tls, [help], [want=$withval], [want=yes])\n",
            "--with-tls",
            Default(True, 1),
        ),
    ],
)
def test_autoconf_literal_declarations_are_accepted(text: str, flag: str, expected: Default) -> None:
    assert _autoconf_default(text, flag) == expected


AUTOCONF_BASE = "AC_ARG_ENABLE(tls, [help], [want=$enableval], [want=no])\n"


@pytest.mark.parametrize(
    "later",
    [
        "eval \"want\"'=yes'\n",
        "builtin eval \"want\"'=yes'\n",
        "source ./platform-defaults\n",
        ". ./platform-defaults\n",
        "read want < platform-defaults\n",
        "export want\n",
        "local want\n",
        "declare want\n",
        "printf -v want yes\n",
        "for want in yes; do :; done\n",
        "unset -v want\n",
        "want=yes\n",
    ],
)
def test_autoconf_indirect_later_writes_abstain(later: str) -> None:
    assert _autoconf_default(AUTOCONF_BASE + later, "--enable-tls") is None


def test_autoconf_function_defined_before_the_declaration_abstains() -> None:
    text = "apply_defaults() {\n  want=yes\n}\n" + AUTOCONF_BASE + "apply_defaults\n"

    assert _autoconf_default(text, "--enable-tls") is None


def test_autoconf_unrelated_later_commands_still_confirm() -> None:
    text = AUTOCONF_BASE + "AC_MSG_CHECKING([done])\necho done\n"

    assert _autoconf_default(text, "--enable-tls") == Default(False, 1)


# --- shell control flow --------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "accepted"),
    [
        ("", True),
        ("if true; then\n", False),
        ("if a; then\ncase x in\ny)\n", False),
        ("for a in b; do\nwhile c; do\n", False),
        ("(\n", False),
        ("{\n", False),
        ("apply() {\n", False),
        ("cat <<EOF\nif true; then\nEOF\n", True),
        ("# if true; then\n", True),
        ('echo "if true; then"\n', True),
        ("echo 'fi'\n", True),
        ("if true; then echo x; fi\n", True),
    ],
)
def test_unconditional_shell_rejects_open_control_flow(prefix: str, accepted: bool) -> None:
    assert _unconditional_shell(prefix + "body\n", len(prefix)) is accepted


# --- input boundary ------------------------------------------------------


def test_parse_default_rejects_bad_text_and_unknown_systems() -> None:
    for text in (CMAKE_BASE + "\x00", CMAKE_BASE + "\ufffd"):
        assert parse_default(BuildSystem.CMAKE, text, "FEATURE") is None
    assert parse_default(BuildSystem.MESON, CMAKE_BASE, "FEATURE") is None


def test_parse_default_rejects_a_flag_that_cannot_be_a_cmake_identifier() -> None:
    for flag in ("", "-FEATURE", "FEATURE;id", "FE ATURE"):
        assert parse_default(BuildSystem.CMAKE, CMAKE_BASE, flag) is None


def test_parse_default_accepts_a_file_at_the_byte_limit() -> None:
    exact = CMAKE_BASE + "#" + " " * (MAX_BUILD_FILE_BYTES - len(CMAKE_BASE) - 2) + "\n"

    assert len(exact.encode("utf-8")) == MAX_BUILD_FILE_BYTES
    assert parse_default(BuildSystem.CMAKE, exact, "FEATURE") == Default(True, 1)


def test_parse_default_rejects_a_file_one_byte_over_the_limit() -> None:
    over = CMAKE_BASE + "#" + " " * (MAX_BUILD_FILE_BYTES - len(CMAKE_BASE) - 1) + "\n"

    assert len(over.encode("utf-8")) == MAX_BUILD_FILE_BYTES + 1
    assert parse_default(BuildSystem.CMAKE, over, "FEATURE") is None
