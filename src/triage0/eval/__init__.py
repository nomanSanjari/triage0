"""Evaluation entry points."""
from .baseline import (
    BaselineError,
    BaselineLabel,
    BaselineVerdict,
    load_reports,
    call_model,
    collect_repository_context,
    resolve_git_ref,
    run_baseline,
)

__all__ = [
    "BaselineError",
    "BaselineLabel",
    "BaselineVerdict",
    "load_reports",
    "call_model",
    "collect_repository_context",
    "resolve_git_ref",
    "run_baseline",
]