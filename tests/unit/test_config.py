"""Tests for ``.env`` loading and its use by the baseline HTTP client."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

from triage0.config import load_env
from triage0.eval.baseline import baseline as baseline_module

_VALID_RESPONSE = {
    "model": "fake-model",
    "choices": [
        {"message": {"content": '{"verdict":"VALID","rationale":"ok"}'}}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


@pytest.fixture
def isolated_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    # python-dotenv writes into os.environ; replace the mapping so loaded keys
    # cannot leak into other tests.
    monkeypatch.setattr(os, "environ", dict(os.environ))


@pytest.fixture
def captured_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> list[urllib.request.Request]:
    requests: list[urllib.request.Request] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return json.dumps(_VALID_RESPONSE).encode("utf-8")

    class CapturingOpener:
        def open(
            self, request: urllib.request.Request, timeout: float | None = None
        ) -> FakeResponse:
            requests.append(request)
            return FakeResponse()

    monkeypatch.setattr(
        baseline_module.urllib.request,
        "build_opener",
        lambda *args, **kwargs: CapturingOpener(),
    )
    return requests


def _write_env(path: Path, **values: str) -> Path:
    env_file = path / ".env"
    env_file.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    return env_file


def test_load_env_reads_explicit_file(
    isolated_environ: None, tmp_path: Path
) -> None:
    env_file = _write_env(tmp_path, TRIAGE0_BASELINE_API_KEY="from-file")

    loaded = load_env(env_file)

    assert loaded == env_file
    assert os.environ["TRIAGE0_BASELINE_API_KEY"] == "from-file"


def test_load_env_reads_default_file_from_working_directory(
    isolated_environ: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = _write_env(tmp_path, TRIAGE0_BASELINE_API_KEY="from-cwd")
    monkeypatch.chdir(tmp_path)

    loaded = load_env()

    assert loaded == env_file
    assert os.environ["TRIAGE0_BASELINE_API_KEY"] == "from-cwd"


def test_load_env_honors_override_variable(
    isolated_environ: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("TRIAGE0_BASELINE_API_KEY=from-custom\n", encoding="utf-8")
    monkeypatch.setenv("TRIAGE0_ENV_FILE", str(env_file))

    loaded = load_env()

    assert loaded == env_file
    assert os.environ["TRIAGE0_BASELINE_API_KEY"] == "from-custom"


def test_load_env_does_not_override_process_environment(
    isolated_environ: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIAGE0_BASELINE_API_KEY", "from-process")
    env_file = _write_env(tmp_path, TRIAGE0_BASELINE_API_KEY="from-file")

    load_env(env_file)

    assert os.environ["TRIAGE0_BASELINE_API_KEY"] == "from-process"


def test_load_env_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert load_env(tmp_path / "missing.env") is None


def test_call_model_loads_api_key_from_dotenv(
    isolated_environ: None,
    captured_requests: list[urllib.request.Request],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRIAGE0_BASELINE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_env(tmp_path, TRIAGE0_BASELINE_API_KEY="dotenv-key")

    result = baseline_module.call_model(
        "prompt", model="fake-model", max_retries=0
    )

    assert result["verdict"] == "VALID"
    assert (
        captured_requests[0].get_header("Authorization") == "Bearer dotenv-key"
    )


def test_process_environment_wins_over_dotenv(
    isolated_environ: None,
    captured_requests: list[urllib.request.Request],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRIAGE0_BASELINE_API_KEY", "process-key")
    _write_env(tmp_path, TRIAGE0_BASELINE_API_KEY="dotenv-key")

    baseline_module.call_model("prompt", model="fake-model", max_retries=0)

    assert (
        captured_requests[0].get_header("Authorization")
        == "Bearer process-key"
    )


def test_explicit_api_key_wins_over_dotenv(
    isolated_environ: None,
    captured_requests: list[urllib.request.Request],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRIAGE0_BASELINE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_env(tmp_path, TRIAGE0_BASELINE_API_KEY="dotenv-key")

    baseline_module.call_model(
        "prompt",
        model="fake-model",
        api_key="explicit-key",
        max_retries=0,
    )

    assert (
        captured_requests[0].get_header("Authorization")
        == "Bearer explicit-key"
    )
