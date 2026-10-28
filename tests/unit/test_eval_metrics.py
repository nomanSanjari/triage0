import pytest

from triage0.eval.metrics import (
    BaselineLabel,
    BaselineVerdict,
    calculate_metrics,
    coerce_baseline_verdict,
    normalize_verdict,
)


def test_metrics_use_expected_denominators():
    rows = [
        {"ground_truth": "fake", "verdict": "INVALID", "cost_usd": 0.01},
        {"ground_truth": "fake", "verdict": "VALID", "cost_usd": 0.02},
        {"ground_truth": "real", "verdict": "INVALID", "cost_usd": 0.01},
        {"ground_truth": "real", "verdict": "UNCERTAIN", "cost_usd": 0.00},
    ]
    metrics = calculate_metrics(rows)
    assert metrics["fake_catch_rate"] == 0.5
    assert metrics["false_flag_rate_on_real"] == 0.5
    assert metrics["cant_check_share"] == 0.25
    assert metrics["cost_per_report_usd"] == 0.01
    assert metrics["fake_catch_count"] == 1
    assert metrics["false_flag_count_on_real"] == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("VALID", BaselineVerdict.VALID),
        ("valid", BaselineVerdict.VALID),
        ("CONFIRMED", BaselineVerdict.VALID),
        ("rejected", BaselineVerdict.INVALID),
        ("LIKELY_INVALID", BaselineVerdict.INVALID),
        ("can't check", BaselineVerdict.UNCERTAIN),
        ("can’t check", BaselineVerdict.UNCERTAIN),
        ("inconclusive", BaselineVerdict.UNCERTAIN),
        ("something unexpected", None),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_coerce_baseline_verdict(raw, expected):
    assert coerce_baseline_verdict(raw) is expected


def test_ground_truth_labels_are_not_model_verdicts():
    assert coerce_baseline_verdict(BaselineLabel.FAKE) is None
    assert coerce_baseline_verdict(BaselineLabel.REAL) is None
    assert normalize_verdict(BaselineLabel.FAKE) == "UNCERTAIN"


def test_unknown_verdict_becomes_uncertain():
    assert normalize_verdict("something unexpected") == "UNCERTAIN"
    assert normalize_verdict("can't check") == "UNCERTAIN"
    assert normalize_verdict("rejected") == "INVALID"


def test_cost_is_null_when_not_known_for_every_report():
    metrics = calculate_metrics([
        {"ground_truth": "fake", "verdict": "INVALID", "cost_usd": 0.01},
        {"ground_truth": "real", "verdict": "VALID", "cost_usd": None},
    ])
    assert metrics["total_cost_usd"] is None
    assert metrics["cost_per_report_usd"] is None
    assert metrics["cost_coverage"] == "1/2"


def test_empty_results_rejected():
    with pytest.raises(ValueError, match="empty"):
        calculate_metrics([])


def test_invalid_label_rejected():
    with pytest.raises(ValueError, match="ground_truth"):
        calculate_metrics([{"ground_truth": "maybe", "verdict": "VALID"}])


@pytest.mark.parametrize("cost", [-0.01, float("nan"), float("inf"), "unknown"])
def test_invalid_cost_rejected(cost):
    with pytest.raises(ValueError, match="cost_usd"):
        calculate_metrics([
            {"ground_truth": "fake", "verdict": "VALID", "cost_usd": cost},
        ])


def test_execution_failures_are_excluded_from_model_abstention_metrics():
    metrics = calculate_metrics([
        {"id": "f1", "ground_truth": "fake", "verdict": None, "error": "timeout", "cost_usd": None},
        {"id": "f2", "ground_truth": "fake", "verdict": "INVALID", "cost_usd": 0.01},
        {"id": "r1", "ground_truth": "real", "verdict": "UNCERTAIN", "cost_usd": 0.02},
    ])
    assert metrics["report_count"] == 3
    assert metrics["evaluated_report_count"] == 2
    assert metrics["execution_error_count"] == 1
    assert metrics["cant_check_count"] == 1
    assert metrics["cant_check_share"] == 0.5
    assert metrics["fake_report_count"] == 2
    assert metrics["evaluated_fake_report_count"] == 1
    assert metrics["fake_catch_rate"] == 1.0


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate report id"):
        calculate_metrics([
            {"id": "same", "ground_truth": "fake", "verdict": "INVALID"},
            {"id": "same", "ground_truth": "real", "verdict": "VALID"},
        ])


def test_row_with_error_and_verdict_rejected():
    with pytest.raises(ValueError, match="both an execution error and a model verdict"):
        calculate_metrics([
            {"ground_truth": "fake", "verdict": "VALID", "error": "timeout"},
        ])


def test_parse_error_row_must_be_uncertain():
    with pytest.raises(ValueError, match="verdict parse error"):
        calculate_metrics([
            {
                "ground_truth": "fake",
                "verdict": "VALID",
                "verdict_parse_error": "unsupported verdict",
            },
        ])


def test_parse_error_abstention_counts_as_uncertain():
    metrics = calculate_metrics([
        {
            "id": "f1",
            "ground_truth": "fake",
            "verdict": "UNCERTAIN",
            "verdict_parse_error": "unsupported verdict",
            "cost_usd": 0.01,
        },
    ])
    assert metrics["cant_check_count"] == 1
    assert metrics["cant_check_share"] == 1.0
    assert metrics["fake_catch_count"] == 0
