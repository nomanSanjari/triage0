import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from triage0.eval import baseline
from triage0.eval.baseline import (
    BaselineError,
    ChatCompletionResponse,
    DatasetReport,
    EvaluationResult,
    ModelPrediction,
    ModelVerdict,
    _parse_model_response,
    build_prompt,
    collect_repository_context,
    load_reports,
    resolve_git_ref,
    run_baseline,
)
from triage0.eval.baseline import baseline as baseline_module
from triage0.eval.metrics import BaselineLabel, BaselineVerdict


def _prediction_payload(**overrides):
    payload = {
        "model": "test-model",
        "verdict": "UNCERTAIN",
        "rationale": "not enough evidence",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_usd": None,
        "latency_seconds": 0.01,
        "raw_response": {"verdict": "UNCERTAIN"},
    }
    payload.update(overrides)
    return payload


def _install_api_response(monkeypatch, payload):
    # Replace HTTP entirely; tests must never make a billable API request.
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    class MockOpener:
        def open(self, request, timeout=None):
            return FakeResponse()

    monkeypatch.setattr(
        baseline_module.urllib.request,
        "build_opener",
        lambda *args, **kwargs: MockOpener(),
    )


def _init_git_repo(path: Path) -> str:
    # Build a disposable repository so revision tests stay offline and isolated.
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "tests@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Tests"],
        check=True,
    )
    (path / "README.md").write_text("repository\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_loads_markdown_reports_by_directory_label(tmp_path):
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    (fake_dir / "report.md").write_text("# Fake report\nA claim.", encoding="utf-8")
    rows = load_reports(fake_dir, "fake")
    assert len(rows) == 1
    assert rows[0]["ground_truth"] == "fake"
    assert "A claim" in rows[0]["report"]


def test_loads_json_records(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "cves.json").write_text(
        json.dumps([{"cve_id": "CVE-2099-0001", "summary": "A bug", "details": "Details"}]),
        encoding="utf-8",
    )
    rows = load_reports(real_dir, "real")
    assert rows[0]["id"] == "CVE-2099-0001"
    assert rows[0]["ground_truth"] == "real"
    assert "Details" in rows[0]["report"]


def test_dataset_report_accepts_valid_label_enum():
    record = DatasetReport(
        id="report-1",
        report="A report",
        ground_truth=BaselineLabel.FAKE,
    )
    assert record.ground_truth is BaselineLabel.FAKE


@pytest.mark.parametrize(
    "record",
    [
        {"id": "", "report": "text", "ground_truth": "fake"},
        {"id": "report-1", "report": "   ", "ground_truth": "fake"},
        {"id": "report-1", "report": "text", "ground_truth": "maybe"},
        {"id": "report-1", "report": "text", "ground_truth": "fake", "extra": True},
    ],
)
def test_dataset_report_rejects_invalid_records(record):
    with pytest.raises(ValidationError):
        DatasetReport.model_validate(record)


def test_load_reports_rejects_invalid_ground_truth(tmp_path):
    with pytest.raises(ValueError, match="ground_truth"):
        load_reports(tmp_path, "maybe")


def test_load_reports_wraps_dataset_schema_errors(tmp_path):
    (tmp_path / "reports.jsonl").write_text(
        json.dumps({"id": "", "report": "text"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BaselineError, match="Invalid dataset record"):
        load_reports(tmp_path, "fake")


def test_load_reports_applies_id_prefix_and_qualifies_duplicates(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    record = json.dumps({"id": "same", "report": "report text"})
    (left / "reports.json").write_text(record, encoding="utf-8")
    (right / "reports.json").write_text(record, encoding="utf-8")

    rows = load_reports(tmp_path, "fake", id_prefix="fake:")

    assert {row["id"] for row in rows} == {
        "fake:left/reports.json:same",
        "fake:right/reports.json:same",
    }


def test_load_reports_rejects_duplicate_ids_within_one_file(tmp_path):
    record = json.dumps({"id": "same", "report": "report text"})
    (tmp_path / "reports.json").write_text(
        json.dumps([json.loads(record), json.loads(record)]),
        encoding="utf-8",
    )
    with pytest.raises(BaselineError, match="duplicate report IDs"):
        load_reports(tmp_path, "fake")


def test_model_verdict_accepts_supported_aliases():
    assert ModelVerdict.model_validate({"verdict": "valid"}).verdict is BaselineVerdict.VALID
    assert ModelVerdict.model_validate({"verdict": "rejected"}).verdict is BaselineVerdict.INVALID
    assert ModelVerdict.model_validate({"verdict": "can't check"}).verdict is BaselineVerdict.UNCERTAIN


def test_model_verdict_rejects_unknown_value_and_accepts_extra_metadata():
    with pytest.raises(ValidationError):
        ModelVerdict.model_validate({"verdict": "maybe"})
    verdict = ModelVerdict.model_validate(
        {"verdict": "VALID", "rationale": "ok", "request_id": "req-1"}
    )
    assert verdict.model_extra == {"request_id": "req-1"}


def test_model_prediction_rejects_parse_error_on_supported_verdict():
    with pytest.raises(ValidationError, match="parse errors"):
        ModelPrediction.model_validate(
            _prediction_payload(
                verdict="VALID",
                verdict_parse_error="unsupported verdict",
            )
        )


def test_model_prediction_normalizes_verdict_aliases():
    prediction = ModelPrediction.model_validate(
        _prediction_payload(verdict="rejected")
    )
    assert prediction.verdict is BaselineVerdict.INVALID


def test_parse_model_response_marks_malformed_output_as_uncertain():
    parsed, parse_error, raw = _parse_model_response("not json")

    assert parsed.verdict is BaselineVerdict.UNCERTAIN
    assert parse_error is not None
    assert raw["raw_text"] == "not json"


def test_parse_model_response_marks_unsupported_verdict_as_uncertain():
    parsed, parse_error, raw = _parse_model_response('{"verdict": "MAYBE"}')

    assert parsed.verdict is BaselineVerdict.UNCERTAIN
    assert parse_error is not None
    assert raw == {"verdict": "MAYBE"}


def test_parse_model_response_rejects_nonstandard_json_constants():
    parsed, parse_error, _ = _parse_model_response(
        '{"verdict": "VALID", "confidence": NaN}'
    )

    assert parsed.verdict is BaselineVerdict.UNCERTAIN
    assert parse_error is not None


def test_parse_model_response_accepts_fenced_json():
    parsed, parse_error, _ = _parse_model_response(
        '```json\n{"verdict": "rejected", "rationale": "contradicted"}\n```'
    )

    assert parsed.verdict is BaselineVerdict.INVALID
    assert parsed.rationale == "contradicted"
    assert parse_error is None


def test_chat_completion_response_validates_envelope_and_normalizes_usage():
    response = ChatCompletionResponse.model_validate({
        "model": "fake-model",
        "choices": [{"message": {"content": "hello"}}],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": "unknown",
            "input_tokens": "7",
            "output_tokens": True,
        },
    })

    assert response.choices[0].message.content == "hello"
    assert response.usage is not None
    assert response.usage.prompt_tokens is None
    assert response.usage.completion_tokens is None
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens is None


def test_chat_completion_response_rejects_missing_choices():
    with pytest.raises(ValidationError):
        ChatCompletionResponse.model_validate({"model": "fake-model", "choices": []})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"verdict": None, "error": None},
        {"verdict": "VALID", "error": "timeout"},
        {"verdict": "VALID", "verdict_parse_error": "bad model output"},
    ],
)
def test_evaluation_result_rejects_inconsistent_verdict_and_error(kwargs):
    payload = {
        "id": "report-1",
        "ground_truth": "fake",
        "repository_revision": "abc123",
    }
    payload.update(kwargs)
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(payload)


def test_evaluation_result_accepts_execution_failure_without_verdict():
    result = EvaluationResult(
        id="report-1",
        ground_truth=BaselineLabel.FAKE,
        repository_revision="abc123",
        error="provider unavailable",
    )
    assert result.verdict is None
    assert result.error == "provider unavailable"


def test_collect_repository_context_is_bounded_and_skips_git(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "secret.md").write_text("ignore me", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello repo", encoding="utf-8")
    context = collect_repository_context(tmp_path, max_context_chars=1000)
    assert "hello repo" in context
    assert "ignore me" not in context


def test_prompt_explicitly_has_no_tools():
    prompt = build_prompt("report text", "repository text")
    assert "NO tools" in prompt
    assert "report text" in prompt
    assert "repository text" in prompt


def test_missing_dataset_has_actionable_error(tmp_path):
    with pytest.raises(BaselineError, match="does not exist"):
        load_reports(tmp_path / "missing", "fake")


def test_expands_osv_vulnerabilities_bundle(tmp_path):
    bundle = tmp_path / "vuln.json"
    bundle.write_text(json.dumps({
        "vulnerabilities": [
            {"id": "CVE-2099-0001", "summary": "Issue A", "details": "Details A",
             "affected": [{"package": {"name": "curl", "ecosystem": "OSS-Fuzz"}}]},
            {"id": "CVE-2099-0002", "summary": "Issue B", "details": "Details B"},
        ]
    }), encoding="utf-8")
    rows = load_reports(tmp_path, "real")
    assert len(rows) == 2
    assert rows[0]["id"] == "CVE-2099-0001"
    assert "Details A" in rows[0]["report"]
    assert "Affected package" in rows[0]["report"]


def test_json_report_captures_git_revision_metadata(tmp_path):
    data = tmp_path / "report.json"
    data.write_text(json.dumps({
        "id": "report-1", "summary": "Example issue", "commit": "abc123"
    }), encoding="utf-8")
    row = load_reports(tmp_path, "real")[0]
    assert row["repo_ref"] == "abc123"


def test_versioned_context_reads_git_commit_not_working_tree(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "tests@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Tests"], check=True)
    source = tmp_path / "src" / "example.c"
    source.parent.mkdir()
    source.write_text("int old_version(void) { return 1; }\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "old"], check=True, capture_output=True)
    old_commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    source.write_text("int new_version(void) { return 2; }\n", encoding="utf-8")
    context = collect_repository_context(tmp_path, ref=old_commit, report="src/example.c")
    assert "old_version" in context
    assert "new_version" not in context


def test_resolve_git_ref_rejects_empty_or_unknown_refs(tmp_path):
    commit = _init_git_repo(tmp_path)

    assert resolve_git_ref(tmp_path, commit) == commit
    with pytest.raises(BaselineError, match="must not be empty"):
        resolve_git_ref(tmp_path, "  ")
    with pytest.raises(BaselineError, match="Git command failed"):
        resolve_git_ref(tmp_path, "does-not-exist")


def test_run_baseline_tracks_resolved_commit(monkeypatch, tmp_path):
    commit = _init_git_repo(tmp_path)
    monkeypatch.setattr(baseline_module, "call_model", lambda *args, **kwargs: _prediction_payload(
        verdict="VALID",
        rationale="supported",
    ))

    result = run_baseline(
        [{"id": "r1", "report": "report", "ground_truth": "real"}],
        repository_path=tmp_path,
        repo_ref="HEAD",
    )

    assert result["results"][0]["repository_version_matched"] is True
    assert result["results"][0]["repository_revision"] == commit
    assert result["resolved_repository_revisions"] == {"HEAD": commit}
    assert "version_match_warning" not in result["results"][0]


def test_run_baseline_marks_prebuilt_context_as_unpinned(monkeypatch):
    prompts: list[str] = []

    def record_prompt(prompt, **kwargs):
        prompts.append(prompt)
        return _prediction_payload()

    monkeypatch.setattr(baseline_module, "call_model", record_prompt)

    result = run_baseline(
        [{"id": "r1", "report": "report", "ground_truth": "real"}],
        repository_context="repository context",
    )

    row = result["results"][0]
    assert row["repository_version_matched"] is False
    assert row["repository_revision"] == "provided-context/unpinned"
    assert "caller-provided repository_context" in row["version_match_warning"]
    assert "repository context" in prompts[0]


def test_run_baseline_warns_when_ref_is_requested_for_prebuilt_context(monkeypatch):
    monkeypatch.setattr(baseline_module, "call_model", lambda *args, **kwargs: _prediction_payload())

    result = run_baseline(
        [
            {
                "id": "r1",
                "report": "report",
                "ground_truth": "real",
                "repo_ref": "abc123",
            }
        ],
        repository_context="repository context",
    )

    row = result["results"][0]
    assert row["repository_version_matched"] is False
    assert row["repository_revision"] == "abc123"
    assert "prebuilt repository_context" in row["version_match_warning"]


def test_run_baseline_qualifies_cross_label_duplicate_ids(monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    monkeypatch.setattr(baseline_module, "call_model", lambda *args, **kwargs: _prediction_payload())

    result = run_baseline(
        [
            {"id": "same", "report": "fake report", "ground_truth": "fake"},
            {"id": "same", "report": "real report", "ground_truth": "real"},
        ],
        repository_path=tmp_path,
    )

    assert {row["id"] for row in result["results"]} == {"fake:same", "real:same"}
    assert result["run_complete"] is True


def test_run_baseline_rejects_invalid_record_before_api_call(monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    monkeypatch.setattr(
        baseline_module,
        "call_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("call_model must not run for an invalid record")
        ),
    )

    with pytest.raises(BaselineError, match="Invalid dataset record"):
        run_baseline(
            [{"id": "f1", "report": "report", "ground_truth": "maybe"}],
            repository_path=tmp_path,
        )


def test_run_baseline_records_malformed_output_as_completed_uncertain(monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    monkeypatch.setattr(
        baseline_module,
        "call_model",
        lambda *args, **kwargs: _prediction_payload(
            verdict="UNCERTAIN",
            rationale="Model returned malformed JSON.",
            verdict_parse_error="Model output failed verdict validation: verdict: invalid",
        ),
    )

    result = run_baseline(
        [{"id": "f1", "report": "report", "ground_truth": "fake"}],
        repository_path=tmp_path,
    )

    assert result["run_complete"] is True
    assert result["metrics"]["execution_error_count"] == 0
    assert result["metrics"]["cant_check_count"] == 1
    assert result["results"][0]["verdict"] == "UNCERTAIN"
    assert result["results"][0]["verdict_parse_error"] is not None


def test_continue_on_error_does_not_count_failure_as_abstention(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("readme", encoding="utf-8")
    monkeypatch.setattr(baseline_module, "call_model", lambda *args, **kwargs: (_ for _ in ()).throw(BaselineError("offline")))
    result = run_baseline(
        [{"id": "f1", "report": "report", "ground_truth": "fake"}],
        repository_path=repo,
        fail_fast=False,
    )
    assert result["run_complete"] is False
    assert result["metrics"]["execution_error_count"] == 1
    assert result["metrics"]["evaluated_report_count"] == 0
    assert result["metrics"]["cant_check_count"] == 0
    assert result["metrics"]["fake_catch_rate"] is None
    assert result["results"][0]["verdict"] is None
    assert result["results"][0]["error"] == "offline"


def test_run_fails_fast_on_execution_error(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("readme", encoding="utf-8")
    monkeypatch.setattr(baseline_module, "call_model", lambda *args, **kwargs: (_ for _ in ()).throw(BaselineError("offline")))
    with pytest.raises(BaselineError, match="failed: offline"):
        run_baseline(
            [{"id": "f1", "report": "report", "ground_truth": "fake"}],
            repository_path=repo,
        )


def test_api_client_retries_rate_limit_and_parses_usage(monkeypatch):
    import io
    import urllib.error
    from http.client import HTTPMessage

    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "model": "fake-model",
                "choices": [{"message": {"content": '{"verdict":"VALID","rationale":"supported"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }).encode()

    class MockOpener:
        def __init__(self):
            self.calls = {"count": 0}

        def open(self, request, timeout=None):
            self.calls["count"] += 1
            if self.calls["count"] == 1:
                raise urllib.error.HTTPError(
                    request.full_url, 429, "rate limited", HTTPMessage(), io.BytesIO(b"try later")
                )
            return FakeResponse()

    mock_opener = MockOpener()
    monkeypatch.setattr(
        baseline_module.urllib.request,
        "build_opener",
        lambda *args, **kwargs: mock_opener,
    )
    monkeypatch.setattr(baseline_module.time, "sleep", lambda _: None)
    monkeypatch.setenv("TRIAGE0_BASELINE_INPUT_USD_PER_1M", "1.0")
    monkeypatch.setenv("TRIAGE0_BASELINE_OUTPUT_USD_PER_1M", "2.0")
    result = baseline_module.call_model("prompt", model="fake-model", api_key="test", max_retries=1)
    assert mock_opener.calls["count"] == 2
    assert result["verdict"] == "VALID"
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 5
    assert result["cost_usd"] == 0.00002


def test_call_model_rejects_invalid_api_envelope(monkeypatch):
    _install_api_response(monkeypatch, {"model": "fake-model"})

    with pytest.raises(BaselineError, match="Unexpected model API response shape"):
        baseline_module.call_model("prompt", model="fake-model", api_key="test")


def test_call_model_rejects_non_text_content(monkeypatch):
    _install_api_response(monkeypatch, {
        "choices": [{"message": {"content": None}}],
    })

    with pytest.raises(BaselineError, match="not text"):
        baseline_module.call_model("prompt", model="fake-model", api_key="test")


def test_call_model_rejects_invalid_message_content_type(monkeypatch):
    _install_api_response(monkeypatch, {
        "choices": [{"message": {"content": 42}}],
    })

    with pytest.raises(BaselineError, match="Unexpected model API response shape"):
        baseline_module.call_model("prompt", model="fake-model", api_key="test")


def test_call_model_rejects_invalid_api_json(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"not json"

    class MockOpener:
        def open(self, request, timeout=None):
            return FakeResponse()

    monkeypatch.setattr(
        baseline_module.urllib.request,
        "build_opener",
        lambda *args, **kwargs: MockOpener(),
    )

    with pytest.raises(BaselineError, match="invalid JSON"):
        baseline_module.call_model("prompt", model="fake-model", api_key="test")


def test_call_model_rejects_nonstandard_json_constants_in_envelope(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return (
                b'{"model": "fake-model", '
                b'"choices": [{"message": {"content": "{\\"verdict\\": \\"VALID\\"}"}}], '
                b'"usage": {"prompt_tokens": NaN}}'
            )

    class MockOpener:
        def open(self, request, timeout=None):
            return FakeResponse()

    monkeypatch.setattr(
        baseline_module.urllib.request,
        "build_opener",
        lambda *args, **kwargs: MockOpener(),
    )

    with pytest.raises(BaselineError, match="invalid JSON"):
        baseline_module.call_model("prompt", model="fake-model", api_key="test")


def test_call_model_normalizes_invalid_usage(monkeypatch):
    _install_api_response(monkeypatch, {
        "model": "fake-model",
        "choices": [{"message": {"content": '{"verdict":"VALID"}'}}],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": "unknown",
            "input_tokens": "7",
            "output_tokens": True,
        },
    })

    result = baseline_module.call_model("prompt", model="fake-model", api_key="test")

    assert result["prompt_tokens"] == 7
    assert result["completion_tokens"] is None
    assert result["cost_usd"] is None


def test_call_model_records_malformed_model_output_as_uncertain(monkeypatch):
    _install_api_response(monkeypatch, {
        "model": "fake-model",
        "choices": [{"message": {"content": "not json"}}],
    })

    result = baseline_module.call_model("prompt", model="fake-model", api_key="test")

    assert result["verdict"] == "UNCERTAIN"
    assert result["verdict_parse_error"] is not None
    assert result["raw_response"]["raw_text"] == "not json"


def test_call_model_requires_api_key(monkeypatch):
    monkeypatch.delenv("TRIAGE0_BASELINE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(BaselineError, match="No API key"):
        baseline_module.call_model("prompt", model="fake-model")


def test_call_model_validates_pricing_before_request(monkeypatch):
    monkeypatch.setenv("TRIAGE0_BASELINE_INPUT_USD_PER_1M", "-1")
    class MockOpener:
        def open(self, request, timeout=None):
            raise AssertionError("no request should be made")

    monkeypatch.setattr(
        baseline_module.urllib.request,
        "build_opener",
        lambda *args, **kwargs: MockOpener(),
    )

    with pytest.raises(BaselineError, match="non-negative"):
        baseline_module.call_model("prompt", model="fake-model", api_key="test")
