"""Metrics and verdict vocabulary for triage0's LLM-only baseline.

These verdicts are benchmark predictions only. They are not triage0 evidence
statuses and must never be used as an operational rejection decision.
"""
from __future__ import annotations

import math
from collections import Counter
from enum import StrEnum
from typing import Any, Iterable, Mapping

__all__ = [
    "BaselineLabel",
    "BaselineVerdict",
    "VALID_LABELS",
    "VALID_VERDICTS",
    "calculate_metrics",
    "coerce_baseline_verdict",
    "compute_metrics",
    "normalize_verdict",
]


class BaselineVerdict(StrEnum):
    """The only verdicts the baseline may emit."""

    VALID = "VALID"
    INVALID = "INVALID"
    UNCERTAIN = "UNCERTAIN"


class BaselineLabel(StrEnum):
    """Ground-truth labels accepted from the evaluation datasets."""

    FAKE = "fake"
    REAL = "real"


VALID_VERDICTS = {verdict.value for verdict in BaselineVerdict}
VALID_LABELS = {label.value for label in BaselineLabel}

# Common prose labels from models, mapped into the closed public vocabulary.
_VERDICT_ALIASES = {
    "TRUE": "VALID",
    "CONFIRMED": "VALID",
    "ACCEPT": "VALID",
    "ACCEPTED": "VALID",
    "LIKELY_VALID": "VALID",
    "REAL": "VALID",
    "FALSE": "INVALID",
    "REJECT": "INVALID",
    "REJECTED": "INVALID",
    "LIKELY_INVALID": "INVALID",
    "FAKE": "INVALID",
    "CANT_CHECK": "UNCERTAIN",
    "CAN'T_CHECK": "UNCERTAIN",
    "CAN_T_CHECK": "UNCERTAIN",
    "CANNOT_CHECK": "UNCERTAIN",
    "INCONCLUSIVE": "UNCERTAIN",
    "UNKNOWN": "UNCERTAIN",
    "UNSURE": "UNCERTAIN",
    # Handle right single quotation mark (U+2019) and other apostrophe variants
    "CAN'T_CHECK": "UNCERTAIN",
    "CANTCHECK": "UNCERTAIN",
    "CANT_CHECK": "UNCERTAIN",  # Already exists but explicit
}


def coerce_baseline_verdict(value: Any) -> BaselineVerdict | None:
    """Map common model-output variants to a supported verdict.

    Returns ``None`` for unknown, non-text, or ground-truth-label values.
    Model-output parsing records that as malformed; metrics treat it as an
    abstention for backward compatibility with older result files.
    """

    if isinstance(value, BaselineVerdict):
        return value
    # Ground-truth labels are not model predictions, even though both are strings.
    if isinstance(value, BaselineLabel) or not isinstance(value, str):
        return None
    # Normalize formatting variants before trying the alias table.
    # Replace right single quotation mark (U+2019) and replacement char (U+FFFD) with apostrophe.
    normalized = value.strip().upper().replace("\u2019", "'").replace("\ufffd", "'").replace("-", "_").replace(" ", "_")
    normalized = _VERDICT_ALIASES.get(normalized, normalized)
    try:
        return BaselineVerdict(normalized)
    except ValueError:
        return None


def normalize_verdict(value: Any) -> str:
    """Normalize common model-output variants; unknown values abstain."""

    verdict = coerce_baseline_verdict(value)
    return verdict.value if verdict is not None else BaselineVerdict.UNCERTAIN.value


def calculate_metrics(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score completed model predictions while reporting execution coverage.

    Rows with ``error`` are execution failures, not model abstentions, and are
    excluded from prediction-rate denominators. They remain in report_count and
    are surfaced in execution_error_count / evaluation_coverage.

    Parse errors (malformed model output recorded as UNCERTAIN with a non-null
    verdict_parse_error) are tracked separately from genuine UNCERTAIN
    abstentions so they can be distinguished in metrics.
    """

    rows = list(results)
    if not rows:
        raise ValueError("Cannot calculate metrics for an empty evaluation set.")

    seen: set[str] = set()
    # Reject ambiguous rows before they can influence any denominator.
    for index, row in enumerate(rows):
        identifier = str(row.get("id", f"row-{index}"))
        if identifier in seen:
            raise ValueError(
                f"Duplicate report id {identifier!r} in evaluation results."
            )
        seen.add(identifier)
        label = str(row.get("ground_truth", "")).strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(
                f"Result {index} has invalid ground_truth {label!r}; "
                "expected 'fake' or 'real'."
            )
        if row.get("error") and row.get("verdict") is not None:
            raise ValueError(
                f"Result {index} has both an execution error and a model verdict."
            )
        if (
            row.get("verdict_parse_error") is not None
            and normalize_verdict(row.get("verdict")) != BaselineVerdict.UNCERTAIN
        ):
            raise ValueError(
                f"Result {index} has a verdict parse error but is not UNCERTAIN."
            )

    errors = [row for row in rows if row.get("error")]
    # Only completed predictions contribute to prediction rates; failures remain
    # visible in report_count and execution_error_count.
    completed = [
        row
        for row in rows
        if not row.get("error") and row.get("verdict") is not None
    ]
    all_fake = [
        row for row in rows if str(row["ground_truth"]).strip().lower() == "fake"
    ]
    all_real = [
        row for row in rows if str(row["ground_truth"]).strip().lower() == "real"
    ]
    fake = [
        row
        for row in completed
        if str(row["ground_truth"]).strip().lower() == "fake"
    ]
    real = [
        row
        for row in completed
        if str(row["ground_truth"]).strip().lower() == "real"
    ]

    # Overall verdict counts (completed predictions only)
    verdict_counts: Counter[str] = Counter(
        normalize_verdict(row.get("verdict")) for row in completed
    )

    # Per-label verdict counts for detailed breakdown
    fake_verdict_counts: Counter[str] = Counter(
        normalize_verdict(row.get("verdict")) for row in fake
    )
    real_verdict_counts: Counter[str] = Counter(
        normalize_verdict(row.get("verdict")) for row in real
    )

    fake_caught = fake_verdict_counts[BaselineVerdict.INVALID.value]
    false_flags = real_verdict_counts[BaselineVerdict.INVALID.value]
    uncertain_total = verdict_counts[BaselineVerdict.UNCERTAIN.value]

    # Parse-error tracking: malformed model outputs recorded as UNCERTAIN
    parse_error_count = sum(
        1 for row in completed if row.get("verdict_parse_error") is not None
    )
    # Genuine uncertainty: UNCERTAIN verdicts without a parse error
    genuine_uncertain_count = uncertain_total - parse_error_count

    total_cost = 0.0
    known_cost_count = 0
    # Aggregate cost only when every report has a usable value.
    for index, row in enumerate(rows):
        cost = row.get("cost_usd")
        if cost is None:
            continue
        try:
            cost_value = float(cost)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Result {index} has invalid cost_usd {cost!r}."
            ) from exc
        if not math.isfinite(cost_value) or cost_value < 0:
            raise ValueError(
                f"Result {index} has invalid cost_usd {cost!r}."
            )
        total_cost += cost_value
        known_cost_count += 1

    total = len(rows)
    evaluated = len(completed)
    fake_count = len(fake)
    real_count = len(real)
    all_fake_count = len(all_fake)
    all_real_count = len(all_real)
    all_costs_known = known_cost_count == total

    return {
        "report_count": total,
        "evaluated_report_count": evaluated,
        "execution_error_count": len(errors),
        "evaluation_coverage": evaluated / total,
        "fake_report_count": all_fake_count,
        "real_report_count": all_real_count,
        "evaluated_fake_report_count": fake_count,
        "evaluated_real_report_count": real_count,
        # Overall verdict distribution
        "verdict_counts": {
            BaselineVerdict.VALID.value: verdict_counts[BaselineVerdict.VALID.value],
            BaselineVerdict.INVALID.value: verdict_counts[BaselineVerdict.INVALID.value],
            BaselineVerdict.UNCERTAIN.value: uncertain_total,
        },
        # Per-label verdict breakdown
        "verdict_counts_by_label": {
            "fake": {
                BaselineVerdict.VALID.value: fake_verdict_counts[BaselineVerdict.VALID.value],
                BaselineVerdict.INVALID.value: fake_verdict_counts[BaselineVerdict.INVALID.value],
                BaselineVerdict.UNCERTAIN.value: fake_verdict_counts[BaselineVerdict.UNCERTAIN.value],
            },
            "real": {
                BaselineVerdict.VALID.value: real_verdict_counts[BaselineVerdict.VALID.value],
                BaselineVerdict.INVALID.value: real_verdict_counts[BaselineVerdict.INVALID.value],
                BaselineVerdict.UNCERTAIN.value: real_verdict_counts[BaselineVerdict.UNCERTAIN.value],
            },
        },
        # Parse-error metrics
        "verdict_parse_error_count": parse_error_count,
        "verdict_parse_error_share": parse_error_count / evaluated if evaluated else None,
        # Genuine uncertainty (abstentions not caused by malformed output)
        "genuine_uncertain_count": genuine_uncertain_count,
        "genuine_uncertain_share": genuine_uncertain_count / evaluated if evaluated else None,
        # Legacy rates (kept for backward compatibility; per-label counts above are preferred)
        "fake_catch_rate": fake_caught / fake_count if fake_count else None,
        "fake_catch_count": fake_caught,
        "false_flag_rate_on_real": false_flags / real_count if real_count else None,
        "false_flag_count_on_real": false_flags,
        # Split cant_check into parse errors vs genuine abstentions
        "cant_check_share": uncertain_total / evaluated if evaluated else None,
        "cant_check_count": uncertain_total,
        "cant_check_parse_error_share": parse_error_count / evaluated if evaluated else None,
        "cant_check_parse_error_count": parse_error_count,
        "cant_check_genuine_abstention_share": genuine_uncertain_count / evaluated if evaluated else None,
        "cant_check_genuine_abstention_count": genuine_uncertain_count,
        "total_cost_usd": round(total_cost, 8) if all_costs_known else None,
        "cost_per_report_usd": round(total_cost / total, 8) if all_costs_known else None,
        "cost_coverage": f"{known_cost_count}/{total}",
        "definitions": {
            "fake_catch_rate": (
                "fake reports predicted INVALID / completed fake predictions"
            ),
            "false_flag_rate_on_real": (
                "real CVEs predicted INVALID / completed real predictions"
            ),
            "fake_report_count / real_report_count": (
                "ground-truth totals in the requested dataset, including "
                "execution failures"
            ),
            "evaluated_fake_report_count / evaluated_real_report_count": (
                "reports with completed model predictions, used as rate "
                "denominators"
            ),
            "cant_check_share": (
                "UNCERTAIN model verdicts / completed predictions; execution "
                "failures excluded"
            ),
            "verdict_parse_error_share": (
                "completed predictions with a verdict_parse_error / completed "
                "predictions; these are malformed model outputs recorded as UNCERTAIN"
            ),
            "genuine_uncertain_share": (
                "UNCERTAIN model verdicts without a parse_error / completed "
                "predictions; genuine model abstentions"
            ),
            "verdict_counts_by_label": (
                "verdict distribution split by ground-truth label (fake/real); "
                "preferred over aggregate rates"
            ),
            "cost": (
                "reported API usage cost; null unless cost is known for every "
                "report"
            ),
        },
    }


compute_metrics = calculate_metrics