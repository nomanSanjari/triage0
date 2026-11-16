"""Run the decomposer and skeptic with a real LLM over the golden fixtures.

The unit tests use a scripted fake, so only this shows what a real model
extracts. It prints extracted vs expected claims per fixture rather than
asserting them: a model's answers vary, and the point is to read them.

Costs money and needs a provider: set TRIAGE0_LLM_BASE_URL, TRIAGE0_LLM_MODEL
and TRIAGE0_LLM_API_KEY (environment or .env), then run
    TRIAGE0_LLM_LIVE_TESTS=1 pytest -s tests/integration/test_extraction_live.py
"""

from __future__ import annotations

import os

import pytest

from tests.golden_fixtures import GoldenCase, load_all
from triage0.core.models import AnyClaim
from triage0.extraction.decomposer import decompose
from triage0.extraction.skeptic import review
from triage0.llm.client import HTTPLLMClient

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAGE0_LLM_LIVE_TESTS") != "1",
    reason="set TRIAGE0_LLM_LIVE_TESTS=1 (calls a paid LLM API)",
)

CASES = {case.case_id: case for case in load_all()}


@pytest.fixture(scope="module")
def client() -> HTTPLLMClient:
    return HTTPLLMClient.from_env()


def _label(claim: AnyClaim) -> str:
    """What a claim is about, short enough to compare by eye."""

    match claim.kind:
        case "symbol":
            return f"symbol {claim.symbol}"
        case "file_line":
            return f"file_line {claim.path}:{claim.line_start}"
        case "call_path":
            return f"call_path {claim.entrypoint} -> {claim.target}"
        case "build_flag":
            return f"build_flag {claim.flag}"
        case "poc":
            return "poc"


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_live_extraction(case: GoldenCase, client: HTTPLLMClient) -> None:
    plan = review(decompose(case.report, case.repo_url, client), case.report, client)

    expected = {_label(e.claim) for e in case.expected_claims}
    print(f"\n== {case.case_id}")
    print(f"   version: expected {case.requested_version!r}, got {plan.requested_version!r}")
    for claim in plan.claims:
        label = _label(claim)
        status = "found " if label in expected else "extra "
        stamp = f"  [skeptic: {claim.result.summary}]" if claim.result else ""
        print(f"   {status} {label}{stamp}")
    for label in sorted(expected - {_label(c) for c in plan.claims}):
        print(f"   missed {label}")
    if plan.notes:
        print(f"   notes: {plan.notes}")
