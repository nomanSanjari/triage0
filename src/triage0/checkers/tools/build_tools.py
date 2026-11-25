"""Conservative readers for literal build-option declarations.

Only the root build file is read, and only inside the sandbox. A regular-file
``find -P`` predicate prevents a checkout symlink from redirecting the read.
These readers describe declaration defaults, not the result of configuring a
project, applying cache overrides, or expanding external modules/macros.

The readers stay deliberately narrow. A CMake file that includes another file
or ends early before the declaration is not analyzed, because either can make
``option()`` a no-op or unreachable. An Autoconf file whose later shell code can
assign the selected variable indirectly (``eval``, ``source``, ``.``, ``read``,
declaration builtins, loops, shell functions) is not analyzed either. Inputs
over :data:`MAX_BUILD_FILE_BYTES` are rejected rather than parsed without bound.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from triage0.core.evidence import Argv
from triage0.core.models import BuildSystem

READ_TIMEOUT_S = 30
# Measured in UTF-8 bytes, so a multi-byte file cannot slip past a character
# count. Oversized files abstain instead of driving unbounded parser work.
MAX_BUILD_FILE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class Default:
    enabled: bool
    line: int


def read_argv(system: BuildSystem) -> Argv:
    path = "/src/CMakeLists.txt" if system is BuildSystem.CMAKE else "/src/configure.ac"
    # -P does not follow a symlink, including one at the starting path. A
    # symlink to another file inside /src is just as unsuitable as one outside.
    return ["find", "-P", path, "-maxdepth", "0", "-type", "f", "-exec", "cat", "--", "{}", "+"]


def parse_default(system: BuildSystem, text: str, flag: str) -> Default | None:
    if "\x00" in text or "\ufffd" in text:
        return None
    if len(text.encode("utf-8")) > MAX_BUILD_FILE_BYTES:
        return None
    if system is BuildSystem.CMAKE:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", flag):
            return None
        return _cmake_default(text, flag)
    if system is BuildSystem.AUTOCONF:
        return _autoconf_default(text, flag)
    return None


def _cmake_commands(text: str) -> list[tuple[str, str, int]] | None:
    """Lex whole commands so comments, strings and bracket literals cannot fake one."""
    commands: list[tuple[str, str, int]] = []
    identifier = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    bracket_open = re.compile(r"\[(=*)\[")
    i = 0
    n = len(text)
    line = 1
    previous = 0

    def bracket(pos: int) -> int | None:
        match = bracket_open.match(text, pos)
        if not match:
            return None
        end = text.find("]" + match.group(1) + "]", pos + len(match.group(0)))
        return n + 1 if end < 0 else end + len(match.group(1)) + 2

    def skip(pos: int) -> int:
        while pos < n:
            if text[pos].isspace():
                pos += 1
            elif text[pos] == "#":
                end = bracket(pos + 1)
                if end is not None:
                    pos = end
                else:
                    endline = text.find("\n", pos)
                    pos = n if endline < 0 else endline + 1
            else:
                break
        return pos

    while True:
        i = skip(i)
        if i == n:
            return commands
        if i > n:
            return None
        name = identifier.match(text, i)
        if name is None:
            return None
        line += text.count("\n", previous, i)
        previous = i
        i = skip(name.end())
        if i >= n or text[i] != "(":
            return None
        body_start = i + 1
        depth = 1
        i += 1
        while i < n and depth:
            char = text[i]
            if char == "#":
                i = skip(i)
                continue
            if char == '"':
                i += 1
                while i < n and text[i] != '"':
                    i += 2 if text[i] == "\\" else 1
                if i >= n:
                    return None
                i += 1
                continue
            end = bracket(i) if char == "[" else None
            if end is not None:
                if end > n:
                    return None
                i = end
                continue
            if char == "\\":
                i += 2
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            i += 1
        if depth:
            return None
        commands.append((name.group(0).lower(), text[body_start:i - 1], line))


def _cmake_default(text: str, flag: str) -> Default | None:
    commands = _cmake_commands(text)
    if commands is None:
        return None
    # Each frame records whether it is a function/macro body: a return() there
    # does not stop the root file.
    stack: list[tuple[str, bool]] = []
    found: list[Default] = []
    uncertain = False
    included = False
    exited = False
    opening = {"if": "endif", "foreach": "endforeach", "while": "endwhile", "function": "endfunction", "macro": "endmacro", "block": "endblock"}
    closing = set(opening.values())
    for name, body, line in commands:
        if name in opening:
            stack.append((opening[name], name in {"function", "macro"}))
        elif name in closing:
            if not stack or stack.pop()[0] != name:
                return None
        # An earlier include() or add_subdirectory() can set the flag as a
        # normal or cache variable, which makes option() do nothing (CMP0077).
        if name in {"include", "add_subdirectory"}:
            included = True
        # return() stops the root file; FATAL_ERROR stops the configure step.
        # Either can leave the declaration unreachable.
        if name == "return" and not any(in_function for _, in_function in stack):
            exited = True
        if name == "message" and re.match(r"\s*FATAL_ERROR\b", body):
            exited = True
        # This command can call option() or set() without spelling either.
        if name == "cmake_language":
            uncertain = True
        if name in {"option", "cmake_dependent_option"}:
            first = re.match(r'\s*(?:"([A-Za-z_][A-Za-z0-9_]*)"|([A-Za-z_][A-Za-z0-9_]*))(?=\s|$)', body)
            if first is None:
                # A dynamic or unsupported spelling might resolve to this flag.
                uncertain = True
                continue
            arg = first.group(1) or first.group(2)
            if arg != flag:
                continue
            match = re.fullmatch(
                r'\s*' + re.escape(first.group(0).strip())
                + r'\s+"(?:[^"\\]|\\.)*"\s+((?i:ON|OFF))\s*',
                body,
            )
            if name != "option" or stack or match is None or included or exited:
                uncertain = True
            else:
                found.append(Default(match.group(1).upper() == "ON", line))
        # Known reassignment and command redefinition forms can change the
        # result even when the option declaration itself is literal.
        if name in {"set", "unset", "set_property", "set_source_files_properties"} and re.search(r"(?<![A-Za-z0-9_])" + re.escape(flag) + r"(?![A-Za-z0-9_])", body):
            uncertain = True
        if name in {"function", "macro"} and re.match(r"\s*option(?:\s|$)", body, re.IGNORECASE):
            uncertain = True
    if stack or uncertain or len(found) != 1:
        return None
    return found[0]


def _m4_call(text: str, opening: int) -> tuple[list[str], int] | None:
    """Split an m4 call at unquoted, top-level commas; preserve nested quotes."""
    args: list[str] = []
    start = opening + 1
    parens = 1
    brackets = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "[":
            brackets += 1
        elif c == "]" and brackets:
            brackets -= 1
        elif not brackets:
            if c == "(":
                parens += 1
            elif c == ")":
                parens -= 1
                if not parens:
                    args.append(text[start:i])
                    return args, i + 1
            elif c == "," and parens == 1:
                args.append(text[start:i])
                start = i + 1
        i += 1
    return None


def _unquote_m4(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return value[1:-1].strip()
    return value


def _autoconf_default(text: str, flag: str) -> Default | None:
    mode: str | None = None
    if flag.startswith("--enable-"):
        mode, flag = "ENABLE", flag[len("--enable-"):]
    elif flag.startswith("--with-"):
        mode, flag = "WITH", flag[len("--with-"):]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", flag):
        return None
    if re.search(r"\b(?:m4_define|m4_defun|AC_DEFUN|define)\s*\(\s*\[?AC_ARG_(?:ENABLE|WITH)\b|\b(?:m4_)?changequote\b", text):
        return None

    # Mask complete macro calls before tracking shell control flow. A nested
    # AC_ARG inside an m4 conditional or definition is never a top-level call.
    masked = list(text)
    calls: list[tuple[str, list[str], int, int, int]] = []
    macro = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    line = 1
    previous = 0
    i = 0
    while i < len(text):
        if text[i] == "#" or (text.startswith("dnl", i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")) and (i + 3 == len(text) or not (text[i + 3].isalnum() or text[i + 3] == "_"))):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
            continue
        if text[i] == "[":
            # m4 square brackets quote text, including apparent macro calls.
            start = i
            depth = 1
            i += 1
            while i < len(text) and depth:
                depth += (text[i] == "[") - (text[i] == "]")
                i += 1
            if depth:
                return None
            for index in range(start, i):
                if masked[index] != "\n":
                    masked[index] = " "
            continue
        if text[i] in "'\"`":
            # Shell quoting does not quote m4, but accepting a macro-looking
            # string here would confuse data with an executed declaration.
            quote = text[i]
            i += 1
            while i < len(text) and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
            if i >= len(text):
                return None
            i += 1
            continue
        if i and (text[i - 1].isalnum() or text[i - 1] == "_"):
            i += 1
            continue
        match = macro.match(text, i)
        if not match:
            i += 1
            continue
        name = match.group(1)
        call = _m4_call(text, match.end() - 1)
        if call is None:
            return None
        args, end = call
        if name in ("AC_ARG_ENABLE", "AC_ARG_WITH"):
            line += text.count("\n", previous, i)
            previous = i
            calls.append((name, args, i, end, line))
        for index in range(i, end):
            if masked[index] != "\n":
                masked[index] = " "
        i = end

    candidates = [c for c in calls if c[1] and _unquote_m4(c[1][0]) == flag and (mode is None or c[0] == "AC_ARG_" + mode)]
    if len(candidates) != 1:
        return None
    name, args, pos, end, line = candidates[0]
    if len(args) != 4:
        return None
    given = _unquote_m4(args[2])
    absent = _unquote_m4(args[3])
    value_name = "enableval" if name == "AC_ARG_ENABLE" else "withval"
    assignment = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=\$" + value_name, given)
    if assignment is None:
        return None
    default = re.fullmatch(re.escape(assignment.group(1)) + r"=(yes|no)", absent)
    if default is None or not _unconditional_shell("".join(masked), pos):
        return None
    # Any later shell construct that can assign the variable makes the
    # declaration default platform-dependent rather than literal.
    if _later_modification(text, end, assignment.group(1)):
        return None
    return Default(default.group(1) == "yes", line)


def _later_modification(text: str, end: int, variable: str) -> bool:
    """Report whether shell code after ``end`` can assign ``variable``.

    This is a rejection list, not an interpreter. It covers the indirect-write
    forms this checker supports; other shell syntax is outside the analyzed
    subset rather than silently interpreted.
    """
    name = re.escape(variable)
    patterns = (
        r"\beval\b",
        r"\bbuiltin\s+(?:eval|source|\.)\b",
        r"(?m:^\s*(?:source|\.)\s+\S)",
        r"[;&|()]\s*(?:source|\.)\s+\S",
        r"\b(?:then|do|else)\s+(?:source|\.)\s+\S",
        rf"\bread\b[^\n]*\b{name}\b",
        rf"\b(?:export|local|declare|typeset|readonly)\b[^\n]*\b{name}\b",
        rf"\bprintf\s+-v\s+{name}\b",
        rf"\bfor\s+{name}\s+in\b",
        rf"\bunset\s+(?:-\w+\s+)*{name}\b",
        rf"\b{name}\s*=",
    )
    if any(re.search(pattern, text[end:]) for pattern in patterns):
        return True
    # A shell function can assign anything, and it may have been defined before
    # the declaration and called after it, so scan the whole file.
    return re.search(r"(?m)^\s*[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{", text) is not None


def _unconditional_shell(text: str, pos: int) -> bool:
    """Accept only a top-level standalone call, outside shell control flow."""
    before = text[:pos]
    if before[before.rfind("\n") + 1:].strip():
        return False
    stack: list[str] = []
    heredoc: str | None = None
    for line in before.splitlines():
        if heredoc is not None:
            if line.strip() == heredoc:
                heredoc = None
            continue
        # Shell strings and comments cannot introduce control keywords.
        line = re.sub(r"'(?:[^']*)'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`", " ", line)
        line = line.split("#", 1)[0]
        line = re.sub(r"\bdnl\b.*", "", line)
        here = re.search(r"<<-?\s*\\?([A-Za-z_][A-Za-z0-9_]*)", line)
        if here:
            heredoc = here.group(1)
        line = re.sub(r"\$\{[^}]*\}", "VAR", line)
        at_command = True
        previous = ""
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[;|&(){}]|[^\s]", line):
            if token in {";", "|", "&"}:
                at_command = True
            elif token == "(" and (at_command or previous == "$"):
                stack.append(")")
                at_command = True
            elif token == ")":
                if stack and stack[-1] == ")":
                    stack.pop()
                # A case-pattern's ')' also starts a command.
                at_command = True
            elif token == "{" and at_command:
                stack.append("}")
            elif token == "}" and at_command:
                if not stack or stack.pop() != "}":
                    return False
                at_command = False
            elif at_command and token in {"if", "case", "for", "while", "until"}:
                stack.append({"if": "fi", "case": "esac", "for": "done", "while": "done", "until": "done"}[token])
                at_command = False
            elif at_command and token in {"fi", "esac", "done"}:
                if not stack or stack.pop() != token:
                    return False
                at_command = False
            elif at_command and token in {"then", "else", "elif", "do"}:
                at_command = True
            else:
                at_command = False
            previous = token
    return not stack and heredoc is None
