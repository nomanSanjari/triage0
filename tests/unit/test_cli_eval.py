import json
from triage0.eval.baseline import baseline as baseline_module

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