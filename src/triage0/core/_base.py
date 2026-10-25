"""Shared Pydantic configuration for triage0 core models.

This module is private.  Import one of the bases below instead of repeating
``ConfigDict`` flags in individual modules.

Pydantic merges ``model_config`` across inheritance, so subclasses only need to
override the settings that differ.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["DomainModel", "EvidenceModel", "FrozenModel"]


class FrozenModel(BaseModel):
    """Strict, immutable base shared by every core schema."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
    )


class EvidenceModel(FrozenModel):
    """Evidence records: preserve command output byte-for-byte."""

    model_config = ConfigDict(str_strip_whitespace=False)


class DomainModel(FrozenModel):
    """Normalized domain objects: trim incidental whitespace."""

    model_config = ConfigDict(str_strip_whitespace=True)
