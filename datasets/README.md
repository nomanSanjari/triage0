# curl ground-truth datasets

Two labelled datasets for the report-verification benchmark, plus the record
schema and the provenance needed to audit or regenerate them.

| Directory | Label | Records | Source |
|---|---|---:|---|
| `curl_slop_49/` | `fake` | 49 | Daniel Stenberg's public `slop.md` list of reports submitted to curl's HackerOne program |
| `curl_cves_239/` | `real` | 239 | curl's official OSV advisory file, `https://curl.se/docs/vuln.json` |

One JSON object per file, named `<id>.json`. There is no bundle file, so
`triage0.eval.baseline.load_reports()` reads each record directly.

## Labels

- **fake** — a report that curl's maintainer listed in `slop.md` as slop
  submitted to the curl bug-bounty program. The label comes from that list,
  never from model judgment.
- **real** — a CVE present in curl's official advisory file. The label comes
  from curl, not from the report text.

The directory a record lives in is authoritative. Each record also carries
`ground_truth` for audit, but the loader uses the directory label.

## Record schema

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable record id (`h1-<digits>` or `CVE-<year>-<digits>`) |
| `case_id` | string | Same value as `id`, for tools that use PR #38's naming |
| `description` | string | One-line human summary; never used as report text |
| `ground_truth` | `"fake"` / `"real"` | Copy of the directory label |
| `repo_url` | URL | `https://github.com/curl/curl` |
| `tag` | string or null | Release tag the commit came from |
| `commit` | string or null | Immutable commit the harness checks out |
| `report` | string | Evaluation input: title, blank line, full public report body |
| `report_id` | string | Source-specific report id |
| `source` | string | `hackerone`, `github_issue`, or `osv_advisory` |
| `project` | string | Always `curl` |
| `title` | string | Report or advisory title |
| `reported_version` | string or null | Version the record is pinned to |
| `source_url` | URL | Public page the report text came from |
| `provenance` | object | Hashes, source metadata, and verification flags |

`provenance` always contains `label_source`, `origin`, `ref_source`,
`ref_verified`, `commit_verified`, `fallback_reason`, and `report_sha256`.
Real records also contain `osv`; report-derived records add `h1` or `github`.

`report_sha256` hashes exactly the stored `report` string, so a reviewer can
confirm the text was not edited after generation.

### Example: real CVE with its original report

```json
{
  "id": "CVE-2026-92391",
  "case_id": "CVE-2026-92391",
  "description": "Real curl CVE CVE-2026-92391; original HackerOne report 4024222; checked at last affected version 8.22.0.",
  "ground_truth": "real",
  "repo_url": "https://github.com/curl/curl",
  "tag": "curl-8_22_0",
  "commit": "01346829096c61b372692f6dc43ffa778c6caccd",
  "report": "CVE-2026-92391: curl_mime_free stack overflow by recursion\n\n<full public body>",
  "report_id": "CVE-2026-92391",
  "source": "hackerone",
  "project": "curl",
  "title": "CVE-2026-92391: curl_mime_free stack overflow by recursion",
  "reported_version": "8.22.0",
  "source_url": "https://hackerone.com/reports/4024222",
  "provenance": { "...": "..." }
}
```

### Example: confirmed fake report

```json
{
  "id": "h1-2199174",
  "case_id": "h1-2199174",
  "description": "Confirmed fake report from bagder/slop.md; HackerOne 2199174.",
  "ground_truth": "fake",
  "repo_url": "https://github.com/curl/curl",
  "tag": "curl-8_4_0",
  "commit": "d755a5f7c009dd63a61b2c745180d8ba937cbfeb",
  "report": "<title>\n\n<full public body>",
  "report_id": "h1-2199174",
  "source": "hackerone",
  "project": "curl",
  "title": "...",
  "reported_version": "8.4.0",
  "source_url": "https://hackerone.com/reports/2199174",
  "provenance": { "...": "..." }
}
```

The 36 fake records whose text names no single resolvable version carry
`tag: null`, `commit: null`, and `reported_version: null` instead.

## Which version is checked

Real records are pinned to `database_specific.last_affected`, never to a
`fixed` version. curl backports fixes, so one advisory can contain several
introduced/fixed pairs:

```
7.56.0 -> 8.14.3, 8.15.0 -> 8.16.2, 8.17.0 -> 8.20.2, 8.21.0 -> 8.23.0
```

`last_affected` is `8.22.0`, so `CVE-2026-92391` is checked at `curl-8_22_0`.
The last fixed version (`8.23.0`) is *not* the check version. All pairs are
preserved in `provenance.osv.ranges`.

`reported_version` is always equal to `provenance.osv.last_affected` for real
records. A test enforces this.

## `tag` and `commit`

`commit` is the contract: the baseline loader reads it as `repo_ref` and
resolves it in the local checkout. `tag` is documentation.

Every record writes both or neither, except the documented tag-gap fallback
below. A tag without a commit would be ignored by the loader and silently
fall back to the working tree, so the generator refuses that combination and a
test enforces it.

For real records the commit is the one the release tag points at, verified
with `git cat-file` against a local curl clone at generation time. In the
current snapshot all 239 real records resolved and verified. For
`CVE-2000-0973` there is no `curl-7_4` tag (only `curl-7_4_1`), so the record
uses the OSV GIT `introduced` commit with `tag: null` and
`ref_source: "introduced_commit_fallback"`.

For fake records, the claimed version is parsed from the report text. A tag is
used only when the text names exactly one version and exactly one release tag
matches it; 13 of the 49 records resolved this way. Otherwise `tag` and
`commit` are both `null` (36 records) and the harness uses its default ref.
This is deliberately conservative: a wrong tag could turn a true claim into a
false `CONTRADICTED`.

## Where the report text comes from

| `source` | Records | Text |
|---|---:|---|
| `hackerone` | 148 real + 49 fake | Public HackerOne report title and `vulnerability_information` |
| `github_issue` | 1 real | Public GitHub issue title and body (`curl/curl#2093`, CVE-2017-8818) |
| `osv_advisory` | 90 real | Generated from the OSV summary, details, ranges, fixed versions, CWE, and advisory URL |

Advisory-derived records set `fallback_reason`:

- `no_original_report_link` — the advisory has no issue link (90 records).
- `h1_report_unavailable` — the linked HackerOne report was not public.
- `github_issue_unavailable` — the linked GitHub issue could not be read.
- `unrecognized_issue_link` — the link was neither HackerOne nor a curl GitHub issue.

None of these were needed for the current snapshot, but the generator records
them honestly if the sources change.

27 advisory-derived records have no GIT range in OSV, so their
`provenance.osv.fix_commits` and `provenance.osv.introduced_commits` are `[]`.
No commit hash is invented.

## Snapshot

Retrieved 2026-10-31.

| Source | SHA-256 | Size |
|---|---|---:|
| `https://curl.se/docs/vuln.json` | `97e81f9a06590abd5641ce14269f613eac26fc109bc7f57aa1df1390110dc164` | 650,967 bytes |
| `https://gist.githubusercontent.com/bagder/07f7581f6e3d78ef37dfbfc81fd1d1cd/raw` | `54f732d5ac0f27662443e907fd0986757ce3a8053eb9a515802f3fad91da5afe` | 6,456 bytes |

The 239 count is current. Older documentation says 215; that number is stale.

## Privacy and attribution

The report bodies are public and are stored verbatim because the benchmark
needs the original text offline. Structured reporter identities, attachments,
bounty amounts, and voter lists are deliberately not copied. Attribution stays
in `source_url` and `provenance`.

The files are research data, not a redistribution license for the underlying
reports. Keep the source URLs when reusing them.

## Regenerating

The builder runs on the host. It refuses redirects, only talks to allowlisted
HTTPS hosts, caps response size, and paces requests serially because
HackerOne returns HTTP 429 under concurrency.

The builder lives in `src/triage0/storage/indexers/`: `osv_advisories` parses
`vuln.json`, `osv_refs` maps versions to tags and verifies commits,
`osv_origins` collects report text, and `osv_sync` coordinates the refresh and
provides this CLI.

By default it creates a temporary bare partial clone of curl
(`--filter=blob:none`) to resolve tags and verify commits. The clone is
removed when the run ends, including after errors and timeouts. Pass
`--curl-repo <path>` to reuse an existing clone instead; the path must be a
directory containing a git repository (normal or bare), and the run fails
fast with a clear error otherwise.

Every commit the run relies on — the selected evaluation commit and the
unique GIT fix/introduced hashes published by OSV — is verified to exist in
the clone. A missing hash is printed as a warning, counted in the summary as
`unverified_commits`, and never replaced or dropped: the record keeps the
official hash with `ref_verified: false`.

Run from the repository root with the project installed
(`pip install -e ".[dev]"`). The `\` continuations are POSIX shell; on
Windows put the command on one line (or use backticks in PowerShell).

```bash
# from the repository root; --out/--slop-out match the defaults
python -m triage0.storage.indexers.osv_sync \
  --out datasets/curl_cves_239 \
  --slop-out datasets/curl_slop_49
```

A full refresh checks for files that the run would not write and fails before
writing anything, so a source change cannot silently leave a mixed snapshot.
`--only CVE-2026-92391` is a targeted overwrite for debugging: it skips that
check and leaves the slop dataset untouched.

Each record is written to a temporary file and replaced atomically, but the
dataset as a whole is not: if a write fails midway through a refresh, some
records will already be new while the rest are still old. Re-run the refresh to
finish it.

Useful flags: `--only CVE-2026-92391`, `--dry-run`, `--vuln-json <path>`,
`--delay`, `--retries`, `--skip-slop`, `--curl-repo <path>`.

Generation takes a few minutes. The committed files are the snapshot; nothing
in the test suite or the runtime reads the network.

## Known limitations

- **Flat `report` string.** `eval/baseline/baseline.py` only reads a string
  `report`. A nested `report` object (the shape used by the golden fixtures in
  PR #38) would be skipped and `description` would silently become the report
  text. Until the loader learns the nested shape, this dataset keeps `report`
  as a string and hoists `report_id`, `source`, `project`, `title`, and
  `source_url` to the top level.
- **`tag` is not read by the loader.** Only `commit` is. Both are always
  written together for humans and for a future loader.
- **`case_id` is not an id source in the loader.** `id` is included for that
  reason.
- **Advisory-derived records are not original reports.** Evaluation results
  should be stratified by `source`.
- **The HackerOne `.json` endpoint is not an official API.** It is a public
  page endpoint and may change; regeneration fails loudly rather than writing
  partial records.
