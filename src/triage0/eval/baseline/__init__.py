"""Facade for baseline evaluation — stable public API.

This module re-exports the public names from :mod:`triage0.eval.baseline.baseline`
so that a future split of the monolithic file into smaller modules can proceed
without breaking imports.
"""

from .baseline import (
    # Models (defined in baseline)
    BaselineError,
    BaselineLabel,
    BaselineVerdict,
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    DatasetReport,
    EvaluationResult,
    ModelPrediction,
    ModelVerdict,
    TokenUsage,
    # Dataset
    load_reports,
    # HTTP
    call_model,
    # Context
    collect_repository_context,
    resolve_git_ref,
    # Runner
    run_baseline,
    # Internal helpers (used by tests)
    build_prompt,
    _parse_model_response,
)

__all__ = [
    "BaselineError",
    "BaselineLabel",
    "BaselineVerdict",
    "ChatChoice",
    "ChatCompletionResponse",
    "ChatMessage",
    "DatasetReport",
    "EvaluationResult",
    "ModelPrediction",
    "ModelVerdict",
    "TokenUsage",
    "load_reports",
    "call_model",
    "collect_repository_context",
    "resolve_git_ref",
    "run_baseline",
    "build_prompt",
    "_parse_model_response",
]