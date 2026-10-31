"""Integrity checks for the committed curl datasets.

These tests read the snapshot in ``datasets/``. They never touch the network:
the JSON files are the authoritative record of what was fetched.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator

from triage0.eval.baseline import load_reports

ROOT = Path(__file__).resolve().parents[3]
SLOP_DIR = ROOT / "datasets" / "curl_slop_49"
CVE_DIR = ROOT / "datasets" / "curl_cves_239"

FORBIDDEN_KEYS = {
    "reporter",
    "attachments",
    "bounty_amount",
    "has_bounty?",
    "voters",
    "vote_count",
}


def records(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]


def keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys(item)


def test_dataset_counts() -> None:
    assert len(list(SLOP_DIR.glob("*.json"))) == 49
    assert len(list(CVE_DIR.glob("*.json"))) == 239


def test_every_record_has_flat_report_text() -> None:
    for path in [*sorted(SLOP_DIR.glob("*.json")), *sorted(CVE_DIR.glob("*.json"))]:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(record["report"], str), path.name
        assert record["report"].strip(), path.name
        assert record["report"] != record["description"], path.name


def test_every_record_uses_reported_version() -> None:
    for directory in (SLOP_DIR, CVE_DIR):
        for record in records(directory):
            assert "reported_version" in record, record["id"]
            assert "requested_version" not in record, record["id"]


def test_report_sha256_matches_the_stored_text() -> None:
    for directory in (SLOP_DIR, CVE_DIR):
        for record in records(directory):
            expected = sha256(record["report"].encode("utf-8")).hexdigest()
            assert record["provenance"]["report_sha256"] == expected, record["id"]


def test_ids_match_file_names_and_are_unique() -> None:
    for directory in (SLOP_DIR, CVE_DIR):
        ids = []
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record["id"] == path.stem
            assert record["case_id"] == record["id"]
            ids.append(record["id"])
        assert len(ids) == len(set(ids))


def test_tag_always_travels_with_commit() -> None:
    for directory in (SLOP_DIR, CVE_DIR):
        for record in records(directory):
            if record["tag"] is not None:
                assert record["commit"] is not None, record["id"]
            elif record["commit"] is not None:
                # A commit without a tag is only allowed for the documented
                # tag-gap fallback, never as a silent substitute.
                assert (
                    record["provenance"]["ref_source"]
                    == "introduced_commit_fallback"
                ), record["id"]


def test_ground_truth_matches_the_directory_label() -> None:
    assert {record["ground_truth"] for record in records(SLOP_DIR)} == {"fake"}
    assert {record["ground_truth"] for record in records(CVE_DIR)} == {"real"}


def test_no_reporter_or_bounty_data_is_stored() -> None:
    for directory in (SLOP_DIR, CVE_DIR):
        for record in records(directory):
            leaked = FORBIDDEN_KEYS.intersection(keys(record))
            assert not leaked, f"{record['id']} contains {sorted(leaked)}"


def test_real_records_pin_a_commit_and_use_last_affected() -> None:
    real = records(CVE_DIR)
    assert all(record["commit"] for record in real)

    without_tag = [record["id"] for record in real if record["tag"] is None]
    assert without_tag == ["CVE-2000-0973"]

    for record in real:
        assert record["reported_version"] == record["provenance"]["osv"]["last_affected"]


def test_cve_2026_92391_is_checked_at_the_last_affected_version() -> None:
    record = json.loads((CVE_DIR / "CVE-2026-92391.json").read_text(encoding="utf-8"))

    assert record["reported_version"] == "8.22.0"
    assert record["tag"] == "curl-8_22_0"
    assert record["commit"] == "01346829096c61b372692f6dc43ffa778c6caccd"
    assert record["commit"] != "d7546809d5bd548b69ec49fca26565c8f6f0d344"


def test_source_mix_is_the_expected_snapshot() -> None:
    sources = [record["source"] for record in records(CVE_DIR)]
    assert sources.count("hackerone") == 148
    assert sources.count("github_issue") == 1
    assert sources.count("osv_advisory") == 90

    fallback = [record for record in records(CVE_DIR) if record["source"] == "osv_advisory"]
    assert all(record["provenance"]["fallback_reason"] for record in fallback)
    assert sum(not record["provenance"]["osv"]["fix_commits"] for record in fallback) == 25


def test_baseline_loader_reads_both_datasets() -> None:
    fake = load_reports(SLOP_DIR, "fake", id_prefix="fake:")
    real = load_reports(CVE_DIR, "real", id_prefix="real:")

    assert len(fake) == 49
    assert len(real) == 239
    combined = fake + real
    assert len({row["id"] for row in combined}) == 288

    for row in real:
        assert row["repo_ref"]
        assert row["ground_truth"] == "real"
    for row in fake:
        assert row["ground_truth"] == "fake"
