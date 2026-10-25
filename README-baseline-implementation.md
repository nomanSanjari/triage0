
# triage0 LLM-only baseline (Task 1.3)

This baseline intentionally does **not** call triage0's extractors, checkers, shell,
repository search, or proof gate. It gives a model the report plus a bounded, read-only
text snapshot of a local checkout in one prompt. Its output is a benchmark prediction,
not a triage0 evidence status or a decision to reject a report.

## Install

From the repository root (Python 3.11+):

```bash
python -m pip install -e ".[dev]"
```

Pydantic 2 is a runtime dependency. The baseline and its tests do not need network
access unless an evaluation is actually run against a model endpoint.

## Configure the model endpoint

The implementation uses the OpenAI-compatible Chat Completions API and Python's standard
library HTTP client. Set:

```powershell
$env:TRIAGE0_BASELINE_API_KEY = "your-api-key"
$env:TRIAGE0_BASELINE_MODEL = "gpt-4.1"
$env:TRIAGE0_BASELINE_REPO = "C:\src\curl"
```

On bash/zsh:

```bash
export TRIAGE0_BASELINE_API_KEY="your-api-key"
export TRIAGE0_BASELINE_MODEL="gpt-4.1"
export TRIAGE0_BASELINE_REPO="$HOME/src/curl"
```

`OPENAI_API_KEY` is also accepted. For an OpenAI-compatible service, set
`TRIAGE0_BASELINE_BASE_URL` to its API base URL or pass `--base-url`.
Model ID and base URL are configurable, so the evaluation logic is not tied to a vendor.

To report cost, set the model's current prices (USD per 1 million tokens):

```bash
TRIAGE0_BASELINE_INPUT_USD_PER_1M="..."
TRIAGE0_BASELINE_OUTPUT_USD_PER_1M="..."
```

Prices are deliberately not hard-coded because they vary by provider and model. If token
usage is unavailable, or prices are not configured, cost metrics are `null` rather than
fabricated.

## Validation architecture

There is no separate schema module. Pydantic 2 models are defined next to the code that
uses them:

- `src/triage0/eval/baseline.py` defines `DatasetReport`, `ModelVerdict`,
  `ModelPrediction`, `EvaluationResult`, and the provider-envelope models
  `ChatCompletionResponse`, `ChatChoice`, `ChatMessage`, and `TokenUsage`.
- `src/triage0/eval/metrics.py` defines the `BaselineVerdict` and `BaselineLabel`
  vocabularies plus verdict alias coercion. Baseline verdicts are restricted to
  `VALID`, `INVALID`, and `UNCERTAIN`.

The validation boundaries behave as follows:

- **Dataset records** must have a non-blank ID, a non-blank report, and ground truth
  `fake` or `real`. Extra record fields are rejected. Invalid records abort loading with
  `BaselineError` and an actionable source location.
- **Model JSON** is parsed with `ModelVerdict`. Supported aliases such as `CONFIRMED`,
  `REJECTED`, and `CANNOT_CHECK` are normalized. Unknown verdicts, missing verdicts,
  non-object JSON, and malformed JSON are recorded as a completed `UNCERTAIN`
  prediction with a non-null `verdict_parse_error`.
- **Provider envelopes** are parsed with `ChatCompletionResponse`. Missing choices,
  invalid message content, non-text content, or invalid JSON are `BaselineError`
  execution failures, not model abstentions.
- **Token usage** is normalized: missing, boolean, negative, or non-numeric counts become
  `null`. Unknown usage does not fail an otherwise valid model response and yields
  `null` cost.
- **Evaluation rows** are parsed with `EvaluationResult`, which requires exactly one of a
  model `verdict` or an execution `error`. Execution failures are excluded from
  prediction-rate denominators and make `run_complete` false. Malformed-output
  abstentions remain completed predictions.

## Dataset layout

By default the CLI expects:

- `datasets/curl_slop_49/`: confirmed fake reports (Markdown, text, JSON, or JSONL)
- `datasets/curl_cves_215/`: real CVE reports (Markdown, text, JSON, or JSONL)

Labels come from which directory a report is in: fake vs. real. Markdown/text files are
one report each. JSON may be one record, a list of records, or `{"reports": [...]}`.
Records should have report prose in `report`, `description`, `body`, `text`, or `content`;
OSV-like records can use `summary` and `details`. JSONL has one record per line.

The uploaded repository scaffold did not include either dataset directory. Add the curated
dataset files before running the full 49-fake / 215-real benchmark.

Within one dataset, duplicate IDs are qualified by relative source path; duplicate IDs
inside one source file are rejected as ambiguous. The CLI additionally prefixes fake IDs
with `fake:` and real IDs with `real:`, so the combined benchmark can contain the same
raw advisory or report ID in both labels without a collision.

## Run

```bash
triage0 eval --baseline --repo C:\src\curl --output results/curl-baseline.json
```

Or set `TRIAGE0_BASELINE_REPO` and run:

```bash
triage0 eval --baseline
```

Use `--fake-dir` and `--real-dir` to override dataset paths, `--repo-ref` to pin reports
without per-report Git revision metadata, and `--limit 2` for a small smoke test.
`--limit` applies after the fake and real datasets are combined, so it selects fake
reports first. The run fails fast by default on API, repository, or dataset errors.
`--continue-on-error` is available only for diagnostic partial runs; failed rows have no
verdict, are excluded from prediction-rate denominators, and cause `run_complete` to be
false with exit code 3.

The output includes a `dataset` block with both loaded counts, the selected count, the
requested limit, and the ID prefixes. Resolved Git references are recorded in
`resolved_repository_revisions`; a report is version-matched only when its requested ref
resolves to a commit in the local checkout.

## Metrics

- `fake_catch_rate`: fake reports predicted `INVALID` / completed fake predictions.
- `false_flag_rate_on_real`: real CVEs predicted `INVALID` / completed real predictions.
- `cant_check_share`: `UNCERTAIN` model verdicts / completed model predictions; execution
  failures are excluded and separately counted.
- `evaluation_coverage`: completed model predictions / all selected reports.
- `cost_per_report_usd`: total reported API cost / all reports. This is `null` unless
  cost can be calculated for every report.

Each record preserves the model verdict, rationale, token usage, latency, source path, and
API errors where applicable. Never use this baseline verdict as a production rejection
signal.

## Verification

The offline test suite covers valid and invalid dataset records, duplicate IDs, supported
and unsupported verdicts, malformed model JSON, API-envelope failures, usage
normalization, result invariants, resolved Git revisions, and CLI limits/error handling.

```bash
python -m pytest -q
python -m mypy
python -m compileall -q src tests
```
