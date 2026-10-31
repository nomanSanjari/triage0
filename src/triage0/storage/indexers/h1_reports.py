"""Acquire public HackerOne reports and turn them into dataset records.

HackerOne renders report pages in JavaScript; ``https://hackerone.com/reports/<id>.json``
returns the same public report as structured data. That endpoint is not an
official API, so every response is treated as untrusted input: the URL is built
from an integer id, redirects are refused, the body is size-capped, and only the
fields used here are validated with Pydantic before a record is written.

Requests are serial on purpose. Four concurrent requests returned HTTP 429
during curation, so pacing and ``Retry-After`` handling are part of the
contract. The committed JSON files are the snapshot; this module exists to
regenerate them, not to run during evaluation.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping
from hashlib import sha256
from http.client import HTTPException, HTTPSConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from triage0.intake._common import normalize_newlines

__all__ = [
    "CURL_HOSTS",
    "CURL_REPO_URL",
    "EXPECTED_SLOP_COUNT",
    "FetchPolicy",
    "GIST_HOSTS",
    "GITHUB_HOSTS",
    "H1_HOSTS",
    "H1Report",
    "H1Error",
    "SLOP_GIST_RAW_URL",
    "SLOP_GIST_URL",
    "VULN_JSON_URL",
    "build_fake_record",
    "fetch_public_json",
    "fetch_public_report",
    "fetch_public_text",
    "parse_slop_list",
    "record_id",
    "report_sha256",
    "report_text",
    "write_json_record",
]

CURL_REPO_URL = "https://github.com/curl/curl"
SLOP_GIST_URL = "https://gist.github.com/bagder/07f7581f6e3d78ef37dfbfc81fd1d1cd"
SLOP_GIST_RAW_URL = (
    "https://gist.githubusercontent.com/bagder/07f7581f6e3d78ef37dfbfc81fd1d1cd/raw"
)
VULN_JSON_URL = "https://curl.se/docs/vuln.json"
H1_REPORT_URL = "https://hackerone.com/reports/{id}.json"
H1_PAGE_URL = "https://hackerone.com/reports/{id}"

H1_TEAM = "curl"
EXPECTED_SLOP_COUNT = 49

H1_HOSTS = frozenset({"hackerone.com"})
GIST_HOSTS = frozenset({"gist.githubusercontent.com"})
CURL_HOSTS = frozenset({"curl.se"})
GITHUB_HOSTS = frozenset({"api.github.com"})

USER_AGENT = "triage0-dataset-builder/0.1 (+https://github.com/nomanSanjari/triage0)"
MAX_RETRY_AFTER_SECONDS = 60.0

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_SLOP_LINK = re.compile(r"https://hackerone\.com/reports/(\d+)")
# Record ids become file names, so they must never contain a path separator.
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class H1Error(RuntimeError):
    """A public report could not be fetched or did not have the expected shape."""


class _Retryable(Exception):
    """A transient HTTP failure that the caller may retry."""

    def __init__(self, message: str, retry_after: float | None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class FetchPolicy(BaseModel):
    """Bounds for one host-side fetch. Shared with :mod:`osv_sync`."""

    model_config = ConfigDict(frozen=True)

    timeout: float = Field(default=30.0, gt=0, le=300)
    retries: int = Field(default=3, ge=0, le=10)
    delay_seconds: float = Field(default=1.2, ge=0, le=60)
    max_bytes: int = Field(default=2_000_000, gt=0, le=50_000_000)


class H1Report(BaseModel):
    """The public parts of one HackerOne report, already validated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    title: str
    body: str
    public: bool
    team_handle: str
    state: str | None = None
    substate: str | None = None
    severity_rating: str | None = None
    weakness: str | None = None
    created_at: str | None = None
    disclosed_at: str | None = None
    cve_ids: tuple[str, ...] = ()

    def provenance(self) -> dict[str, Any]:
        """Describe the report without copying reporter, attachment, or bounty data."""

        return {
            "id": self.id,
            "state": self.state,
            "substate": self.substate,
            "severity_rating": self.severity_rating,
            "weakness": self.weakness,
            "created_at": self.created_at,
            "disclosed_at": self.disclosed_at,
            "public": self.public,
            "cve_ids": list(self.cve_ids),
        }


class _RawReport(BaseModel):
    """The subset of a HackerOne JSON payload this module depends on."""

    model_config = ConfigDict(frozen=True, extra="allow")

    id: int
    title: str
    vulnerability_information: str = ""
    public: bool = False
    team: dict[str, Any] | None = None
    state: str | None = None
    substate: str | None = None
    severity_rating: str | None = None
    weakness: Any = None
    created_at: str | None = None
    disclosed_at: str | None = None
    cve_ids: list[Any] = Field(default_factory=list)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _weakness_text(value: Any) -> str | None:
    """HackerOne sends a CWE object here; keep the readable name, not its repr."""

    if isinstance(value, Mapping):
        for key in ("name", "id"):
            text = _optional_text(value.get(key))
            if text:
                return text
        return None
    return _optional_text(value)


def _error_summary(exc: ValidationError) -> str:
    # Never echo input values: the payload is untrusted report text.
    return "; ".join(
        f"{'.'.join(str(part) for part in error.get('loc', ())) or '<root>'}: "
        f"{error.get('msg', 'invalid value')}"
        for error in exc.errors()[:4]
    )


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    # A non-finite value would reach time.sleep() and raise ValueError.
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


def _https_get(url: str, policy: FetchPolicy, allowed_hosts: frozenset[str]) -> bytes:
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise H1Error(f"refusing non-HTTPS URL: {url}")
    host = parts.hostname
    if host is None or host not in allowed_hosts:
        raise H1Error(f"refusing host {host!r} for {url}")
    # Fixed public endpoints never need credentials; reject userinfo instead
    # of silently dropping it.
    if parts.username is not None or parts.password is not None:
        raise H1Error("refusing URL with embedded credentials")
    try:
        port = parts.port
    except ValueError:
        raise H1Error("refusing URL with an invalid port") from None
    # An explicit port would connect somewhere other than the public endpoint.
    if port not in (None, 443):
        raise H1Error(f"refusing non-standard HTTPS port {port} for {url}")

    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"

    connection = HTTPSConnection(host, parts.port, timeout=policy.timeout)
    try:
        connection.request(
            "GET",
            target,
            headers={
                "Accept": "application/json, text/plain;q=0.9",
                "Accept-Encoding": "identity",
                "User-Agent": USER_AGENT,
            },
        )
        response = connection.getresponse()
        if response.status in _REDIRECT_STATUSES:
            raise H1Error(f"refusing redirect from {url} (HTTP {response.status})")
        if response.status == 429 or 500 <= response.status < 600:
            raise _Retryable(
                f"HTTP {response.status} from {url}",
                _retry_after(response.getheader("Retry-After")),
            )
        if response.status != 200:
            raise H1Error(f"HTTP {response.status} from {url}")
        body = response.read(policy.max_bytes + 1)
        if len(body) > policy.max_bytes:
            raise H1Error(f"response from {url} exceeds {policy.max_bytes} bytes")
        return body
    finally:
        connection.close()


def _fetch_with_retries(
    url: str, policy: FetchPolicy, allowed_hosts: frozenset[str]
) -> bytes:
    last_error = ""
    for attempt in range(policy.retries + 1):
        try:
            return _https_get(url, policy, allowed_hosts)
        except _Retryable as exc:
            last_error = str(exc)
            if attempt >= policy.retries:
                break
            pause = exc.retry_after if exc.retry_after is not None else policy.delay_seconds
            time.sleep(max(pause, 0.0))
        except (OSError, HTTPException) as exc:
            # DNS, connection, TLS and socket failures surface as OSError;
            # truncated or malformed responses as HTTPException. Both are
            # transport problems, so retry and then report H1Error. Without
            # this they would escape the origin fallback and crash the CLI.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= policy.retries:
                break
            time.sleep(max(policy.delay_seconds, 0.0))
    raise H1Error(
        f"giving up on {url} after {policy.retries + 1} attempts: {last_error}"
    )


def fetch_public_json(
    url: str, policy: FetchPolicy, *, allowed_hosts: frozenset[str]
) -> Any:
    """Fetch one allowlisted HTTPS URL and decode it as JSON."""

    payload = _fetch_with_retries(url, policy, allowed_hosts)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise H1Error(f"invalid JSON from {url}: {exc}") from exc


def fetch_public_text(
    url: str, policy: FetchPolicy, *, allowed_hosts: frozenset[str]
) -> str:
    """Fetch one allowlisted HTTPS URL as UTF-8 text."""

    payload = _fetch_with_retries(url, policy, allowed_hosts)
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise H1Error(f"response from {url} is not UTF-8") from exc


def fetch_public_report(report_id: int, policy: FetchPolicy) -> H1Report:
    """Fetch and validate one public curl report from HackerOne."""

    if isinstance(report_id, bool) or not isinstance(report_id, int) or report_id <= 0:
        raise H1Error(f"invalid HackerOne report id: {report_id!r}")
    payload = fetch_public_json(
        H1_REPORT_URL.format(id=report_id), policy, allowed_hosts=H1_HOSTS
    )
    return _report_from_payload(payload, report_id)


def _report_from_payload(payload: Any, expected_id: int) -> H1Report:
    if not isinstance(payload, Mapping):
        raise H1Error(f"HackerOne report {expected_id} is not a JSON object")
    try:
        raw = _RawReport.model_validate(dict(payload))
    except ValidationError as exc:
        raise H1Error(
            f"HackerOne report {expected_id} has an unexpected shape: {_error_summary(exc)}"
        ) from exc

    if raw.id != expected_id:
        raise H1Error(
            f"HackerOne report {expected_id} returned a different id ({raw.id})"
        )
    if not raw.public:
        raise H1Error(f"HackerOne report {expected_id} is not public")
    team = raw.team.get("handle") if raw.team else None
    if team != H1_TEAM:
        raise H1Error(f"HackerOne report {expected_id} is not in the curl program")

    title = normalize_newlines(raw.title).strip()
    body = normalize_newlines(raw.vulnerability_information).strip()
    if not title:
        raise H1Error(f"HackerOne report {expected_id} has an empty title")
    if not body:
        raise H1Error(f"HackerOne report {expected_id} has an empty body")

    cve_ids = tuple(
        text
        for value in raw.cve_ids
        if (text := _optional_text(value)) is not None
    )
    return H1Report(
        id=raw.id,
        title=title,
        body=body,
        public=raw.public,
        team_handle=team,
        state=_optional_text(raw.state),
        substate=_optional_text(raw.substate),
        severity_rating=_optional_text(raw.severity_rating),
        weakness=_weakness_text(raw.weakness),
        created_at=_optional_text(raw.created_at),
        disclosed_at=_optional_text(raw.disclosed_at),
        cve_ids=cve_ids,
    )


def parse_slop_list(gist_text: str) -> list[int]:
    """Extract the 49 confirmed slop report ids from Daniel Stenberg's gist.

    A different count means the source changed; failing here is better than
    silently building a dataset that no longer matches the label.
    """

    links = [int(match.group(1)) for match in _SLOP_LINK.finditer(gist_text)]
    unique: list[int] = []
    seen: set[int] = set()
    for report_id in links:
        if report_id in seen:
            continue
        seen.add(report_id)
        unique.append(report_id)
    if len(links) != len(unique):
        raise H1Error(f"slop list repeats report ids: {len(links)} links, {len(unique)} unique")
    if len(unique) != EXPECTED_SLOP_COUNT:
        raise H1Error(
            f"expected {EXPECTED_SLOP_COUNT} slop reports, found {len(unique)}"
        )
    return unique


def record_id(report_id: int) -> str:
    return f"h1-{report_id}"


def report_text(title: str, body: str) -> str:
    """Build the text the harness evaluates: title, blank line, full body.

    The baseline loader only reads a string ``report`` field, so the title is
    folded in here. A body that already starts with the title is left alone to
    avoid duplicating it; the title must be followed by whitespace so a body
    such as "A buggy parser..." is not mistaken for a title match. Newlines
    are normalized before hashing so ``report_sha256`` stays stable across
    CRLF and LF sources.
    """

    clean_title = normalize_newlines(title).strip()
    clean_body = normalize_newlines(body).strip()
    if not clean_body:
        raise H1Error("report body is empty")
    if clean_title:
        starts_with_title = clean_body == clean_title or clean_body.startswith(
            (f"{clean_title}\n", f"{clean_title} ")
        )
        if not starts_with_title:
            return f"{clean_title}\n\n{clean_body}"
    return clean_body


def report_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def build_fake_record(
    report: H1Report,
    *,
    tag: str | None,
    commit: str | None,
    reported_version: str | None,
    commit_verified: bool | None = None,
) -> dict[str, Any]:
    """Assemble one flat dataset record for a confirmed fake report.

    ``tag`` and ``commit`` travel together: the baseline loader ignores
    ``tag`` and only reads ``commit``, so a tag without a commit would
    silently fall back to the working tree.
    """

    if (tag is None) != (commit is None):
        raise H1Error("tag and commit must be set together")
    if (tag is not None) != (reported_version is not None):
        raise H1Error("a resolved ref needs the version it was resolved from")

    identifier = record_id(report.id)
    text = report_text(report.title, report.body)
    # A tag without a verification result is still unverified; never claim it.
    ref_verified = commit_verified if commit is not None else None
    return {
        "id": identifier,
        "case_id": identifier,
        "description": f"Confirmed fake report from bagder/slop.md; HackerOne {report.id}.",
        "ground_truth": "fake",
        "repo_url": CURL_REPO_URL,
        "tag": tag,
        "commit": commit,
        "report": text,
        "report_id": identifier,
        "source": "hackerone",
        "project": "curl",
        "title": report.title,
        "reported_version": reported_version,
        "source_url": H1_PAGE_URL.format(id=report.id),
        "provenance": {
            "label_source": SLOP_GIST_URL,
            "origin": "hackerone",
            "ref_source": "report_version" if tag is not None else "unresolved",
            "ref_verified": ref_verified,
            "commit_verified": ref_verified,
            "fallback_reason": None,
            "report_sha256": report_sha256(text),
            "h1": report.provenance(),
        },
    }


def write_json_record(directory: Path, record: Mapping[str, Any]) -> Path:
    """Write one record as ``<id>.json`` with stable formatting.

    The write goes to a temporary file in the same directory and is then
    replaced atomically, so an interrupted refresh cannot truncate an existing
    record.
    """

    identifier = record.get("id")
    if not isinstance(identifier, str) or not _SAFE_ID.fullmatch(identifier):
        raise H1Error(f"invalid record id: {identifier!r}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identifier}.json"
    temporary = directory / f".{identifier}.json.tmp"
    payload = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
