"""Command-line entry point for triage0.

``verify`` is functional as far as intake: it reads a report, picks an adapter
(auto-detected or explicit), normalizes it, and renders the result.  Extraction
and sandbox verification are not wired yet, so it says so instead of pretending
to verify.

``eval`` and ``index`` are argument-complete scaffolds.  They exit with
``EXIT_NOT_IMPLEMENTED`` rather than ``0`` so a script cannot mistake an
unimplemented milestone for a successful run.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from triage0 import __version__
from triage0.core.models import NormalizedReport, ReportSource
from triage0.intake import IntakeError, adapter_for, detect_source

EXIT_INTAKE_ERROR = 1
EXIT_NOT_IMPLEMENTED = 2

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Deterministic evidence checker for AI-written bug reports.",
)


class SourceChoice(StrEnum):
    """Explicit intake format, or ``auto`` to sniff the payload."""

    auto = "auto"
    markdown = "markdown"
    email = "email"
    hackerone = "hackerone"
    github = "github"


class OutputFormat(StrEnum):
    """How ``verify`` renders the normalized report."""

    terminal = "terminal"
    json = "json"


_SOURCE_MAP: dict[SourceChoice, ReportSource] = {
    SourceChoice.markdown: ReportSource.MARKDOWN,
    SourceChoice.email: ReportSource.EMAIL,
    SourceChoice.hackerone: ReportSource.HACKERONE,
    SourceChoice.github: ReportSource.GITHUB_ADVISORY,
}


def _resolve_source(choice: SourceChoice, payload: bytes) -> ReportSource:
    if choice is SourceChoice.auto:
        return detect_source(payload)
    source = _SOURCE_MAP.get(choice)
    if source is None:
        raise IntakeError(f"unsupported report source {choice.value!r}")
    return source


def _summary(report: NormalizedReport, repo: str | None) -> str:
    submitted = (
        report.submitted_at.isoformat() if report.submitted_at else "-"
    )
    lines = [
        f"report:  {report.report_id}",
        f"source:  {report.source.value}",
        f"title:   {report.title or '-'}",
        f"project: {report.project or '-'}",
        f"version: {report.reported_version or '-'}",
        f"date:    {submitted}",
        f"url:     {report.source_url or '-'}",
        (
            "body:    "
            f"{len(report.body.splitlines())} lines, "
            f"{len(report.body)} chars"
        ),
        f"hash:    {report.content_hash}",
    ]
    if repo:
        lines.append(f"repo:    {repo}")
    return "\n".join(lines)


def _render(
    report: NormalizedReport, output_format: OutputFormat, repo: str | None
) -> str:
    if output_format is OutputFormat.json:
        return report.model_dump_json(indent=2)
    return _summary(report, repo)


def _emit(text: str, out: Path | None) -> None:
    if out is None:
        typer.echo(text)
        return
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        typer.secho(
            f"triage0: error: could not write {out}: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_INTAKE_ERROR) from exc
    typer.secho(f"Wrote normalized report to {out}", err=True)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"triage0 {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Deterministic evidence checking for AI-written bug reports."""


@app.command()
def verify(
    report: Annotated[
        Path,
        typer.Option(
            "--report",
            "-r",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to the incoming report.",
        ),
    ],
    repo: Annotated[
        str | None,
        typer.Option(
            "--repo",
            help="Target repository, recorded for the verification stage.",
        ),
    ] = None,
    source: Annotated[
        SourceChoice,
        typer.Option(
            "--source",
            help="Input format; auto sniffs Markdown, HackerOne, and GitHub.",
        ),
    ] = SourceChoice.auto,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Rendering of the normalized report."),
    ] = OutputFormat.terminal,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the result here instead of stdout."),
    ] = None,
) -> None:
    """Normalize a bug report and show what verification will receive."""

    try:
        payload = report.read_bytes()
        resolved = _resolve_source(source, payload)
        adapter = adapter_for(resolved)
        forced = None if source is SourceChoice.auto else resolved
        normalized = adapter.normalize(payload, forced_source=forced)
        if repo is not None:
            normalized.metadata["repo"] = repo
    except IntakeError as exc:
        typer.secho(
            f"triage0: error: {exc}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=EXIT_INTAKE_ERROR) from exc
    except OSError as exc:
        typer.secho(
            f"triage0: error: could not read {report}: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_INTAKE_ERROR) from exc

    _emit(_render(normalized, output_format, repo), out)
    typer.secho(
        "intake complete; extraction and verification are not wired yet.",
        fg=typer.colors.YELLOW,
        err=True,
    )


@app.command("eval")
def eval_suite(
    suite: Annotated[
        str,
        typer.Option("--suite", help="Benchmark suite to replay, e.g. curl."),
    ],
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", help="Dataset directory override."),
    ] = None,
    include_baseline: Annotated[
        bool,
        typer.Option(
            "--include-baseline",
            help="Also run the LLM-only baseline.",
        ),
    ] = False,
) -> None:
    """Replay a labelled benchmark suite (not implemented yet)."""

    details = [f"suite={suite}"]
    if dataset is not None:
        details.append(f"dataset={dataset}")
    if include_baseline:
        details.append("baseline=on")
    typer.secho(
        f"triage0: eval is not implemented yet ({', '.join(details)}).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=EXIT_NOT_IMPLEMENTED)


@app.command()
def index(
    osv: Annotated[
        bool,
        typer.Option("--osv", help="Sync the OSV advisory index."),
    ] = False,
    vectors: Annotated[
        bool,
        typer.Option("--vectors", help="Rebuild the report embedding index."),
    ] = False,
    db: Annotated[
        Path | None,
        typer.Option("--db", help="Database path override."),
    ] = None,
) -> None:
    """Build local advisory and embedding indexes (not implemented yet)."""

    requested = [
        name
        for name, enabled in (("osv", osv), ("vectors", vectors))
        if enabled
    ]
    selection = ", ".join(requested) if requested else "all"
    target = f" db={db}" if db is not None else ""
    typer.secho(
        f"triage0: index is not implemented yet ({selection}{target}).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=EXIT_NOT_IMPLEMENTED)


if __name__ == "__main__":
    app()
