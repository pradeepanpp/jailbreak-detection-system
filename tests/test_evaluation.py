# tests/test_evaluation.py
"""
Metric helpers (no models), then sanity checks on the report written by
scripts/evaluate.py (skipped until that report exists).
"""

from types import SimpleNamespace

import evaluate
from src.jailbreak_detection.constants import LABEL_NAMES


def fake_results(*decisions):
    return [SimpleNamespace(decision=d) for d in decisions]


def test_block_and_monitor_rates_for_benign():
    m = evaluate.compute_binary_metrics(
        ["benign", "benign", "benign", "benign", "direct_jailbreak"],
        fake_results("BLOCK", "MONITOR", "ALLOW", "ALLOW", "BLOCK"),
    )
    assert m["benign_block_rate"] == 0.25
    assert m["benign_monitor_rate"] == 0.25
    assert m["false_positive_rate"] == 0.5
    assert m["recall"] == 1.0


def test_wilson_interval():
    low, high = evaluate.wilson_interval(9, 10)
    assert 0.55 < low < 0.6 and 0.98 < high <= 1.0
    assert evaluate.wilson_interval(0, 0) == [0.0, 0.0]


# Report from scripts/evaluate.py

def test_report_has_all_sections(report):
    for key in ["meta", "binary_metrics", "multiclass_metrics", "latency", "ablation", "score_distribution"]:
        assert key in report
    assert report["meta"]["test_size"] > 0
    assert set(report["ablation"]) == {"rule_only", "rule_ml", "full_system"}


def test_binary_metrics_are_consistent(report):
    b = report["binary_metrics"]
    for key in ["accuracy", "precision", "recall", "f1", "false_positive_rate"]:
        assert 0.0 <= b[key] <= 1.0
    assert b["tp"] + b["tn"] + b["fp"] + b["fn"] == report["meta"]["test_size"]


def test_multiclass_metrics_are_consistent(report):
    m = report["multiclass_metrics"]
    assert set(LABEL_NAMES) <= set(m["per_class"])
    matrix = m["confusion_matrix"]["matrix"]
    assert len(matrix) == 7 and all(len(row) == 7 for row in matrix)
    for row, label in zip(matrix, m["confusion_matrix"]["labels"]):
        assert sum(row) == m["per_class"][label]["support"]


def test_latency_is_plausible(report):
    lat = report["latency"]
    assert 0 < lat["min_ms"] <= lat["p50_ms"] <= lat["p95_ms"] <= lat["max_ms"]
    assert lat["mean_ms"] < 5000


def test_quality_gates(report):
    """Regression gates: fail loudly if a retrained model is clearly worse."""
    b = report["binary_metrics"]
    ab = report["ablation"]
    if report["meta"]["layers_active"] == 3:
        assert b["f1"] >= 0.50
    assert b["false_positive_rate"] <= 0.30
    assert ab["full_system"]["f1"] >= ab["rule_only"]["f1"]


def test_attacks_score_higher_than_benign(report):
    dist = report["score_distribution"]
    attack_means = [dist[c]["mean"] for c in LABEL_NAMES if c != "benign"]
    assert dist["benign"]["mean"] < sum(attack_means) / len(attack_means)
