"""Command-line entry point for triage0's LLM-only baseline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from triage0.eval.baseline import BaselineError, load_reports, run_baseline

# Namespace IDs at the CLI boundary, where the two datasets are combined.
FAKE_ID_PREFIX = "fake:"
REAL_ID_PREFIX = "real:"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage0",
        description="Evidence checking for vulnerability reports.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("eval", help="Run evaluation benchmarks.")
    evaluation.add_argument(
        "--baseline", action="store_true",
        help="Run the naive LLM-only baseline (no triage0 verification tools).",
    )
    evaluation.add_argument("--fake-dir", default="datasets/curl_slop_49", help="Confirmed fake reports directory.")
    evaluation.add_argument("--real-dir", default="datasets/curl_cves_215", help="Real CVE reports directory.")
    evaluation.add_argument(
        "--repo", default=os.getenv("TRIAGE0_BASELINE_REPO"),
        help="Local checkout of the target repository (required).",
    )
    evaluation.add_argument(
        "--repo-ref", default=os.getenv("TRIAGE0_BASELINE_REPO_REF"),
        help="Git commit/tag/branch to use for reports without their own revision metadata.",
    )
    evaluation.add_argument("--model", default=os.getenv("TRIAGE0_BASELINE_MODEL"), help="Model ID.")
    evaluation.add_argument(
        "--base-url", default=os.getenv("TRIAGE0_BASELINE_BASE_URL"),
        help="Optional OpenAI-compatible API base URL.",
    )
    evaluation.add_argument("--limit", type=int, default=None, help="Evaluate only the first N reports (useful for smoke tests).")
    evaluation.add_argument("--output", default="-", help="Output JSON path (default: stdout).")
    evaluation.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue after per-report failures, marking the run incomplete. Default is fail-fast.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "eval":
        parser.error("Unknown command.")
    if not args.baseline:
        parser.error("This milestone implements `triage0 eval --baseline` only.")
    if not args.repo:
        parser.error("A local repository checkout is required: pass --repo PATH or set TRIAGE0_BASELINE_REPO.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    try:
        # The prefixes keep IDs unique when both datasets legitimately reuse an
        # advisory or report identifier. They are also recorded in the output.
        fake_reports = load_reports(
            args.fake_dir, "fake", id_prefix=FAKE_ID_PREFIX
        )
        real_reports = load_reports(
            args.real_dir, "real", id_prefix=REAL_ID_PREFIX
        )
        reports = fake_reports + real_reports
        # Apply the limit after concatenation so it controls the actual run size.
        if args.limit is not None:
            reports = reports[:args.limit]
        payload = run_baseline(
            reports,
            repository_path=args.repo,
            repo_ref=args.repo_ref,
            model=args.model,
            base_url=args.base_url,
            fail_fast=not args.continue_on_error,
        )
        # Keep full load counts separate from the subset actually evaluated.
        payload["dataset"] = {
            "fake_dir": str(args.fake_dir),
            "real_dir": str(args.real_dir),
            "fake_reports_loaded": len(fake_reports),
            "real_reports_loaded": len(real_reports),
            "reports_selected_for_run": len(reports),
            "limit": args.limit,
            "id_prefixes": {"fake": FAKE_ID_PREFIX, "real": REAL_ID_PREFIX},
        }
    except (BaselineError, ValueError, OSError) as exc:
        print(f"triage0: error: {exc}", file=sys.stderr)
        return 2

    try:
        # Strict JSON output: NaN/Infinity are not valid benchmark results.
        rendered = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        print(f"triage0: error: could not serialize results: {exc}", file=sys.stderr)
        return 2
    if args.output == "-":
        print(rendered)
    else:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"triage0: error: could not write {args.output}: {exc}", file=sys.stderr)
            return 2
        print(f"Wrote baseline results to {output_path}", file=sys.stderr)
    if not payload["run_complete"]:
        # Exit 3 distinguishes a partial run from a configuration/API error (2).
        print(
            f"triage0: warning: run incomplete; {payload['execution_error_count']} report(s) failed. "
            "See the output JSON; do not treat this as a complete benchmark.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
