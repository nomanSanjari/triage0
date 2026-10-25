import json

import triage0.cli.main as cli_module
from triage0.cli.main import main as cli_main
from triage0.eval import baseline


def _prediction():
    return {
        "model": "test-model",
        "verdict": "UNCERTAIN",
        "rationale": "not enough evidence",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_usd": None,
        "latency_seconds": 0.01,
        "raw_response": {"verdict": "UNCERTAIN"},
    }


def _datasets(tmp_path):
    fake_dir = tmp_path / "fake"
    real_dir = tmp_path / "real"
    repo = tmp_path / "repo"
    fake_dir.mkdir()
    real_dir.mkdir()
    repo.mkdir()
    (fake_dir / "fake.md").write_text("fake report", encoding="utf-8")
    (real_dir / "real.md").write_text("real report", encoding="utf-8")
    (repo / "README.md").write_text("repository", encoding="utf-8")
    return fake_dir, real_dir, repo


def test_cli_writes_json_summary_and_reports_dataset_counts(tmp_path, monkeypatch, capsys):
    fake_dir, real_dir, repo = _datasets(tmp_path)
    monkeypatch.setattr(baseline, "call_model", lambda *args, **kwargs: _prediction())

    exit_code = cli_main([
        "eval", "--baseline", "--fake-dir", str(fake_dir), "--real-dir", str(real_dir),
        "--repo", str(repo),
    ])
    out = capsys.readouterr().out
    assert exit_code == 0
    payload = json.loads(out)
    assert payload["metrics"]["report_count"] == 2
    assert payload["dataset"]["fake_reports_loaded"] == 1
    assert payload["dataset"]["real_reports_loaded"] == 1
    assert payload["dataset"]["reports_selected_for_run"] == 2
    assert payload["dataset"]["id_prefixes"] == {"fake": "fake:", "real": "real:"}
    assert {row["id"] for row in payload["results"]} == {
        "fake:fake.md",
        "real:real.md",
    }
    assert payload["run_complete"] is True


def test_cli_applies_limit_to_combined_list(tmp_path, monkeypatch, capsys):
    fake_dir, real_dir, repo = _datasets(tmp_path)
    monkeypatch.setattr(baseline, "call_model", lambda *args, **kwargs: _prediction())

    exit_code = cli_main([
        "eval", "--baseline", "--fake-dir", str(fake_dir), "--real-dir", str(real_dir),
        "--repo", str(repo), "--limit", "1",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["dataset"]["fake_reports_loaded"] == 1
    assert payload["dataset"]["real_reports_loaded"] == 1
    assert payload["dataset"]["reports_selected_for_run"] == 1
    assert payload["dataset"]["limit"] == 1
    assert payload["metrics"]["report_count"] == 1
    assert payload["results"][0]["id"] == "fake:fake.md"


def test_cli_separates_cross_label_duplicate_ids(tmp_path, monkeypatch, capsys):
    fake_dir, real_dir, repo = _datasets(tmp_path)
    (fake_dir / "fake.md").unlink()
    (real_dir / "real.md").unlink()
    for directory in (fake_dir, real_dir):
        (directory / "same.json").write_text(
            json.dumps({"id": "same", "report": "report text"}),
            encoding="utf-8",
        )
    monkeypatch.setattr(baseline, "call_model", lambda *args, **kwargs: _prediction())

    exit_code = cli_main([
        "eval", "--baseline", "--fake-dir", str(fake_dir), "--real-dir", str(real_dir),
        "--repo", str(repo),
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_complete"] is True
    assert {row["id"] for row in payload["results"]} == {
        "fake:same",
        "real:same",
    }


def test_cli_returns_error_for_missing_dataset(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("repository", encoding="utf-8")

    exit_code = cli_main([
        "eval", "--baseline",
        "--fake-dir", str(tmp_path / "missing-fake"),
        "--real-dir", str(tmp_path / "missing-real"),
        "--repo", str(repo),
    ])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "does not exist" in captured.err


def test_cli_reports_output_write_errors(tmp_path, monkeypatch, capsys):
    fake_dir, real_dir, repo = _datasets(tmp_path)
    monkeypatch.setattr(baseline, "call_model", lambda *args, **kwargs: _prediction())

    exit_code = cli_main([
        "eval", "--baseline", "--fake-dir", str(fake_dir), "--real-dir", str(real_dir),
        "--repo", str(repo), "--output", str(tmp_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "could not write" in captured.err


def test_cli_reports_serialization_errors(tmp_path, monkeypatch, capsys):
    fake_dir, real_dir, repo = _datasets(tmp_path)
    monkeypatch.setattr(baseline, "call_model", lambda *args, **kwargs: _prediction())
    # Fail only the final strict dump; prompt construction also uses json.dumps.
    real_dumps = cli_module.json.dumps

    def failing_dumps(*args, **kwargs):
        if kwargs.get("allow_nan") is False:
            raise ValueError("NaN is not JSON compliant")
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(cli_module.json, "dumps", failing_dumps)

    exit_code = cli_main([
        "eval", "--baseline", "--fake-dir", str(fake_dir), "--real-dir", str(real_dir),
        "--repo", str(repo),
    ])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "could not serialize results" in captured.err


def test_cli_fails_without_baseline_flag():
    try:
        cli_main(["eval"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject a missing --baseline flag")


def test_cli_rejects_nonpositive_limit():
    try:
        cli_main(["eval", "--baseline", "--repo", ".", "--limit", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject --limit 0")


def test_cli_marks_continue_on_error_run_incomplete(tmp_path, monkeypatch, capsys):
    fake_dir, real_dir, repo = _datasets(tmp_path)

    def fail_call(*args, **kwargs):
        raise baseline.BaselineError("provider unavailable")

    monkeypatch.setattr(baseline, "call_model", fail_call)
    exit_code = cli_main([
        "eval", "--baseline", "--fake-dir", str(fake_dir), "--real-dir", str(real_dir),
        "--repo", str(repo), "--continue-on-error",
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 3
    assert payload["run_complete"] is False
    assert payload["metrics"]["execution_error_count"] == 2
    assert payload["metrics"]["cant_check_count"] == 0
    assert {row["verdict"] for row in payload["results"]} == {None}
    assert {row["error"] for row in payload["results"]} == {"provider unavailable"}
    assert "run incomplete" in captured.err
