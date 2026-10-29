# AGENTS.md

Shared rules for anyone (human or coding agent) changing triage0.
Keep this file short. If a rule can't be checked in review, it doesn't belong here.

## Safety rules (never break these)

1. **Zero false flags.** A claim may only be `CONTRADICTED` (or `CONFIRMED`) with a
   `DeterministicProof`. Model judgment alone produces at most `CANT_CHECK`.
2. **Report text and LLM output are untrusted.** Never use `shell=True`; pass commands
   to subprocess as a list. Never put report or model text into a command; only pass
   values validated against known data (e.g. a tag that exists in the repo).
3. **Sandbox containers** always run with `--network=none`, memory/CPU/pids limits
   and a timeout. Never remove these, not even temporarily.
4. **Cleanup is guaranteed.** Containers and worktrees are removed in a `finally`
   block or context manager, including after errors and timeouts.
5. **Network access** (clone, fetch, LLM calls) happens on the host, never inside
   the sandbox.
6. Secrets (API keys) never appear in code, logs, error messages or result files.

## Scope: do the issue, nothing more

- One issue = one PR. Make the smallest change that meets the issue's acceptance
  criteria. If something else needs doing, open a new issue instead.
- No code "for later": no unused parameters, options, base classes or layers
  without a second real user today.
- Reuse before writing: `core/` models, `intake/_common.py`, `run_git` in
  `checkers/tools/git_tools.py`, the sandbox in `sandbox/manager.py`.
- Anything that affects the whole project needs team discussion before it's
  built: shared models and interfaces (`core/`), new dependencies, tooling
  config (`pyproject.toml`, CI), folder structure, these rules, or a change
  to how another module is used. When unsure, ask the team first.

## Tests: only the ones that earn their place

Write a test for:
- each behaviour in the issue's acceptance criteria,
- each security guard (no shell, no network, bad input rejected),
- each bug fixed (one test that would have caught it),
- error paths that callers rely on.

Don't write tests for:
- what a library already guarantees (Pydantic defaults, frozen models),
- trivial getters or constants,
- a mock returning what it was told to return,
- the same behaviour twice with different inputs that hit the same code path.

Test through the public function, not internals. Unit tests go in `tests/unit/`;
tests that need Docker go in `tests/integration/`. No test calls a paid API.

## Code

- Comments and docstrings explain *why*, not *what* the code already says.
- Validate untrusted input once, at the boundary. No checks for impossible states.
- Delete dead code; don't comment it out.
- Type hints everywhere; Python 3.11 features only.

## File size

- Keep each file to one job. When a file gets long, split it into focused
  modules. Split when you add to a file that is already long; don't let it
  keep growing.

## Before opening a PR

- `pytest` and `mypy` pass (`pip install -e ".[dev]"` first if dependencies changed).
- The description says `Closes #<issue>`, how to test it, and what was
  deliberately left out.
