"""Environment loading for triage0.

Values already present in the process environment always take precedence over
entries in a ``.env`` file. Loading is explicit so importing the package does
not mutate global state on its own.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE_NAME = ".env"
ENV_FILE_OVERRIDE = "TRIAGE0_ENV_FILE"


def load_env(path: str | Path | None = None) -> Path | None:
    """Load a ``.env`` file without overriding existing environment values.

    Args:
        path: Explicit environment file. Defaults to ``TRIAGE0_ENV_FILE`` when
            set, otherwise ``.env`` in the current working directory.

    Returns:
        The path that was loaded, or ``None`` when no file exists.
    """

    if path is not None:
        candidate = Path(path).expanduser()
    else:
        configured = os.getenv(ENV_FILE_OVERRIDE) or ENV_FILE_NAME
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate

    if not candidate.is_file():
        return None
    load_dotenv(candidate, override=False)
    return candidate
