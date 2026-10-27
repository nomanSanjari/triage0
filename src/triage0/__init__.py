"""triage0: deterministic evidence checking for bug reports."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("triage0")
except PackageNotFoundError:
    # Running from a source checkout without an editable install.
    __version__ = "0.1.0"

__all__ = ["__version__"]
