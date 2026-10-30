# Golden fixtures

Small curl reports with the correct verdict for every claim in them, checked
by hand against the real curl source at a stated tag. #8 (decomposer and
skeptic), #10 (checkers), #11 (code agent) and #12 (scorer and replay) all
test against the same answers.

They are not the benchmark datasets (#2). Those label whole reports as real or
fake. These label each claim and include edge cases on purpose.

## Using them

```python
from tests.golden_fixtures import load_all

for case in load_all():
    case.report               # a triage0.core.models.NormalizedReport
    for expected in case.expected_claims:
        expected.claim            # a real SymbolClaim, FileLineClaim, ...
        expected.expected_status  # an EvidenceStatus
```

`load_all()` rejects a fixture that doesn't match the format, so a broken file
fails the tests instead of confusing a later one.

## Format

One JSON file per case. The file name is the `case_id`.

| Field | Meaning |
|---|---|
| `case_id` | Same as the file name, without `.json`. |
| `description` | What the case is for and who uses it. |
| `repo_url` | The repository the claims are about. |
| `tag`, `commit` | The version the labels were checked at, and the commit that tag points to. Tags can move; the commit can't. |
| `report` | The report, as `NormalizedReport` fields. |
| `requested_version` | The version text a correct decomposer copies from the report, or `null` if the report gives none. |
| `expected_claims` | The claims a correct decomposer extracts. Each has `claim` (a core claim, chosen by `kind`), `expected_status` and `evidence`. |
| `evidence` | What justifies the label, each with an optional `note`. `"kind": "line"`: `path`, `line` and the exact `text` of that line at `commit`. `"kind": "absent"`: a `pattern` that appears nowhere in `scope`, which is `tree` (any file at `commit`) or `history` (any commit up to it). |

Every claim's `source_excerpt` must appear in the report body, because the
decomposer drops claims whose excerpt it can't find.

## Reading a label

A label is **the true answer at that commit**, provable from the source. It is
not always what one stage of the pipeline can output:

- A stage may return `CANT_CHECK` where the label is `CONFIRMED` or
  `CONTRADICTED`. That is a miss, not a false flag. For example, a static
  call-graph checker can't follow function pointers.
- A stage must **never** return the opposite verdict: `CONTRADICTED` where the
  label is `CONFIRMED`, or the reverse.
- `CANT_CHECK` labels mean the answer can't be proved from the source, so
  `CANT_CHECK` is the only correct output.

**Pending:** labels on claims about something that isn't there (a made-up
symbol, a renamed symbol, a file that only exists after a build) depend on how
checkers signal "not found" to the code agent. That is still being agreed on
#10, and these labels may change once it is.

## Adding a case

1. Pick what the case is for (a claim kind, an edge case) and write the report
   text yourself, or quote a public source and set `report.source_url`. Nothing
   embargoed.
2. Look up every label in the real source at the tag. **Never label from
   memory.**

   ```bash
   git clone --bare https://github.com/curl/curl.git /tmp/curl.git
   git -C /tmp/curl.git rev-parse 'curl-8_23_0^{commit}'          # -> commit
   git -C /tmp/curl.git show curl-8_23_0:lib/curlx/dynbuf.c | sed -n '115p'
   ```

3. Copy each evidence line exactly, including leading spaces.
4. Save it as `<case_id>.json` and run `pytest tests/unit/test_golden_fixtures.py`.
