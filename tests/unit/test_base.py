"""Tests for the shared core model configuration in ``core._base``."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError, field_validator

from triage0.core._base import DomainModel, EvidenceModel, FrozenModel
from triage0.core.evidence import (
    CommandResult,
    DeterministicProof,
    EvidenceResult,
)
from triage0.core.models import NormalizedReport, VerificationPlan


class _EvidenceProbe(EvidenceModel):
    value: str


class _DomainProbe(DomainModel):
    value: str


def test_all_core_models_share_the_frozen_base() -> None:
    for model in (
        CommandResult,
        DeterministicProof,
        EvidenceResult,
        NormalizedReport,
        VerificationPlan,
    ):
        assert issubclass(model, FrozenModel)


def test_config_is_merged_across_inheritance() -> None:
    for base in (EvidenceModel, DomainModel):
        assert base.model_config["frozen"] is True
        assert base.model_config["extra"] == "forbid"
        assert base.model_config["validate_default"] is True

    assert EvidenceModel.model_config["str_strip_whitespace"] is False
    assert DomainModel.model_config["str_strip_whitespace"] is True


def test_extra_fields_rejected_in_both_bases() -> None:
    with pytest.raises(ValidationError):
        _EvidenceProbe(value="x", unexpected="nope")
    with pytest.raises(ValidationError):
        _DomainProbe(value="x", unexpected="nope")


def test_both_bases_are_frozen() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        _EvidenceProbe(value="x").value = "y"
    with pytest.raises(ValidationError, match="frozen"):
        _DomainProbe(value="x").value = "y"


def test_evidence_models_preserve_whitespace() -> None:
    result = CommandResult(
        tool="git",
        argv=("git", "status"),
        exit_code=0,
        stdout="  kept exactly  \n",
        stderr="  error text  \n",
    )
    assert result.stdout == "  kept exactly  \n"
    assert result.stderr == "  error text  \n"


def test_domain_models_strip_whitespace() -> None:
    report = NormalizedReport(
        report_id="r1",
        source="markdown",
        title="  trimmed  ",
        body="  body  ",
    )
    assert report.title == "trimmed"
    assert report.body == "body"


def test_validate_default_is_inherited() -> None:
    class NaiveDefaultProbe(EvidenceModel):
        when: datetime = datetime(2026, 1, 1)

        @field_validator("when")
        @classmethod
        def _must_be_aware(cls, value: datetime) -> datetime:
            if value.tzinfo is None:
                raise ValueError("naive datetime rejected")
            return value

    with pytest.raises(ValidationError, match="naive datetime rejected"):
        NaiveDefaultProbe()
