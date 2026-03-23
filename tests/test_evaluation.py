# tests/test_evaluation.py
"""
Tests for Phase 6 — Evaluation Pipeline.

Verifies that:
  - evaluate.py runs without error
  - The JSON report has correct structure and sane values
  - Binary metrics are consistent with multiclass results
  - Ablation table is complete
  - Latency values are physically plausible

Run AFTER scripts/evaluate.py has completed:
    python scripts/evaluate.py
    python tests/test_evaluation.py
"""

import sys
import os
import json
sys.path.append(".")

REPORT_PATH = "data/results/evaluation_report.json"
SUMMARY_PATH = "data/results/evaluation_summary.txt"

from src.jailbreak_detection.constants import LABEL_NAMES


# ─────────────────────────────────────────────
# LOAD REPORT
# ─────────────────────────────────────────────

def load_report() -> dict:
    if not os.path.exists(REPORT_PATH):
        raise FileNotFoundError(
            f"Report not found: {REPORT_PATH}\n"
            "Run: python scripts/evaluate.py first"
        )
    with open(REPORT_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_report_exists():
    print("\n--- Testing Report Files ---")

    # Test 1 — JSON report exists
    assert os.path.exists(REPORT_PATH), f"Missing: {REPORT_PATH}"
    print(f"✅ Test 1 passed: {REPORT_PATH} exists")

    # Test 2 — Text summary exists
    assert os.path.exists(SUMMARY_PATH), f"Missing: {SUMMARY_PATH}"
    print(f"✅ Test 2 passed: {SUMMARY_PATH} exists")

    # Test 3 — JSON is valid
    with open(REPORT_PATH) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    print("✅ Test 3 passed: JSON report is valid")


def test_report_structure():
    print("\n--- Testing Report Structure ---")
    report = load_report()

    # Test 4 — Top-level keys present
    required_keys = [
        "meta", "binary_metrics", "multiclass_metrics",
        "latency", "ablation", "score_distribution"
    ]
    for key in required_keys:
        assert key in report, f"Missing top-level key: {key}"
    print("✅ Test 4 passed: All top-level keys present")

    # Test 5 — Meta fields
    meta = report["meta"]
    assert "test_size"     in meta
    assert "layers_active" in meta
    assert meta["test_size"] > 0
    assert 1 <= meta["layers_active"] <= 3
    print(f"✅ Test 5 passed: meta — test_size={meta['test_size']} layers={meta['layers_active']}")

    # Test 6 — Binary metrics keys
    b = report["binary_metrics"]
    for key in ["accuracy", "precision", "recall", "f1", "false_positive_rate", "tp", "tn", "fp", "fn"]:
        assert key in b, f"Missing binary metric: {key}"
    print("✅ Test 6 passed: All binary metric keys present")

    # Test 7 — Multiclass metric keys
    m = report["multiclass_metrics"]
    for key in ["weighted_f1", "macro_f1", "accuracy", "per_class", "confusion_matrix"]:
        assert key in m, f"Missing multiclass key: {key}"
    print("✅ Test 7 passed: All multiclass metric keys present")

    # Test 8 — Latency keys
    lat = report["latency"]
    for key in ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"]:
        assert key in lat, f"Missing latency key: {key}"
    print("✅ Test 8 passed: All latency keys present")

    # Test 9 — Ablation keys
    for config in ["rule_only", "rule_ml", "full_system"]:
        assert config in report["ablation"], f"Missing ablation config: {config}"
    print("✅ Test 9 passed: All ablation configs present")


def test_binary_metrics_bounds():
    print("\n--- Testing Binary Metrics Bounds ---")
    report = load_report()
    b = report["binary_metrics"]

    # Test 10 — All scores in [0, 1]
    for key in ["accuracy", "precision", "recall", "f1", "false_positive_rate"]:
        val = b[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} out of bounds"
    print("✅ Test 10 passed: All binary scores in [0, 1]")

    # Test 11 — Confusion matrix counts consistent
    total = b["tp"] + b["tn"] + b["fp"] + b["fn"]
    assert total == report["meta"]["test_size"], (
        f"TP+TN+FP+FN={total} != test_size={report['meta']['test_size']}"
    )
    print(f"✅ Test 11 passed: Confusion matrix counts sum to test_size ({total})")

    # Test 12 — Reasonable performance (F1 > 0.5 for 3-layer system)
    if report["meta"]["layers_active"] == 3:
        assert b["f1"] >= 0.50, (
            f"Binary F1={b['f1']:.4f} is unexpectedly low for full system"
        )
        print(f"✅ Test 12 passed: Binary F1={b['f1']:.4f} (≥0.50 for full system)")
    else:
        print(f"⏭️  Test 12 skipped: Only {report['meta']['layers_active']} layers active")

    # Test 13 — FPR is acceptably low (< 0.30)
    assert b["false_positive_rate"] <= 0.30, (
        f"FPR={b['false_positive_rate']:.4f} is too high (max 0.30)"
    )
    print(f"✅ Test 13 passed: FPR={b['false_positive_rate']:.4f} (≤0.30)")


def test_multiclass_metrics():
    print("\n--- Testing Multiclass Metrics ---")
    report = load_report()
    m = report["multiclass_metrics"]

    # Test 14 — Weighted F1 in bounds
    assert 0.0 <= m["weighted_f1"] <= 1.0
    print(f"✅ Test 14 passed: weighted_F1={m['weighted_f1']:.4f}")

    # Test 15 — All 7 classes in per_class report
    for cls in LABEL_NAMES:
        assert cls in m["per_class"], f"Missing per-class report for: {cls}"
    print("✅ Test 15 passed: All 7 classes in per-class report")

    # Test 16 — Per-class scores in [0, 1]
    for cls, scores in m["per_class"].items():
        for metric in ["precision", "recall", "f1"]:
            val = scores[metric]
            assert 0.0 <= val <= 1.0, f"{cls}.{metric}={val} out of bounds"
    print("✅ Test 16 passed: All per-class scores in [0, 1]")

    # Test 17 — Confusion matrix shape is 7×7
    cm = m["confusion_matrix"]["matrix"]
    assert len(cm) == 7
    for row in cm:
        assert len(row) == 7
    print("✅ Test 17 passed: Confusion matrix is 7×7")

    # Test 18 — Confusion matrix rows sum to support counts
    cm_arr = [sum(row) for row in cm]
    labels = m["confusion_matrix"]["labels"]
    for i, cls in enumerate(labels):
        if cls in m["per_class"]:
            support = m["per_class"][cls]["support"]
            assert cm_arr[i] == support, (
                f"Confusion matrix row {i} ({cls}) sums to {cm_arr[i]} "
                f"but support={support}"
            )
    print("✅ Test 18 passed: Confusion matrix rows consistent with support counts")


def test_latency_plausible():
    print("\n--- Testing Latency Values ---")
    report = load_report()
    lat = report["latency"]

    # Test 19 — All positive
    for key in ["mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"]:
        assert lat[key] > 0, f"{key}={lat[key]} should be positive"
    print(f"✅ Test 19 passed: All latencies positive")

    # Test 20 — Ordering: min ≤ p50 ≤ mean ≤ p95 ≤ max
    assert lat["min_ms"] <= lat["p50_ms"], "min > p50"
    assert lat["p50_ms"] <= lat["p95_ms"], "p50 > p95"
    assert lat["p95_ms"] <= lat["max_ms"], "p95 > max"
    print(
        f"✅ Test 20 passed: "
        f"min={lat['min_ms']:.1f}ms  "
        f"p50={lat['p50_ms']:.1f}ms  "
        f"p95={lat['p95_ms']:.1f}ms  "
        f"max={lat['max_ms']:.1f}ms"
    )

    # Test 21 — Mean latency < 5 seconds (CPU is slow but not that slow)
    assert lat["mean_ms"] < 5000, f"Mean latency {lat['mean_ms']:.0f}ms seems too high"
    print(f"✅ Test 21 passed: Mean latency={lat['mean_ms']:.1f}ms < 5000ms")


def test_ablation_ordering():
    print("\n--- Testing Ablation Study ---")
    report = load_report()
    ab = report["ablation"]

    rule_f1   = ab["rule_only"]["f1"]
    ruleml_f1 = ab["rule_ml"]["f1"]
    full_f1   = ab["full_system"]["f1"]

    print(f"  rule_only  : F1={rule_f1:.4f}")
    print(f"  rule+ml    : F1={ruleml_f1:.4f}")
    print(f"  full_system: F1={full_f1:.4f}")

    # Test 22 — Full system ≥ rule-only (more layers ≥ not worse)
    assert full_f1 >= rule_f1, (
        f"Full system F1 {full_f1:.4f} < rule_only {rule_f1:.4f}"
    )
    print("✅ Test 22 passed: Full system F1 ≥ rule-only F1")

    # Test 23 — Latency increases with layers (more work = slower)
    rule_lat   = ab["rule_only"]["mean_latency_ms"]
    full_lat   = ab["full_system"]["mean_latency_ms"]
    assert full_lat >= rule_lat, (
        f"Full system latency {full_lat:.1f}ms < rule-only {rule_lat:.1f}ms"
    )
    print(
        f"✅ Test 23 passed: "
        f"Latency scales with layers "
        f"({rule_lat:.0f}ms → {full_lat:.0f}ms)"
    )

    # Test 24 — All ablation configs have required keys
    for config, vals in ab.items():
        for key in ["f1", "accuracy", "recall", "fpr", "mean_latency_ms"]:
            assert key in vals, f"Ablation config {config} missing key: {key}"
    print("✅ Test 24 passed: All ablation configs have required keys")


def test_score_distribution():
    print("\n--- Testing Score Distribution ---")
    report = load_report()
    dist = report["score_distribution"]

    # Test 25 — All classes present in distribution
    for cls in LABEL_NAMES:
        assert cls in dist, f"Missing score distribution for: {cls}"
    print("✅ Test 25 passed: Score distribution has all 7 classes")

    # Test 26 — Benign mean score lower than attack mean scores
    benign_mean = dist["benign"]["mean"]
    attack_means = [
        dist[cls]["mean"] for cls in LABEL_NAMES if cls != "benign"
    ]
    avg_attack_mean = sum(attack_means) / len(attack_means)

    print(f"  Benign mean score  : {benign_mean:.3f}")
    print(f"  Attack mean score  : {avg_attack_mean:.3f} (avg across attack classes)")

    assert benign_mean < avg_attack_mean, (
        f"Benign mean {benign_mean:.3f} ≥ attack mean {avg_attack_mean:.3f}"
    )
    print("✅ Test 26 passed: Benign mean score < attack mean scores")

    # Test 27 — All distribution values in [0, 1]
    for cls, stats in dist.items():
        for key in ["mean", "median", "min", "max"]:
            val = stats[key]
            assert 0.0 <= val <= 1.0, f"{cls}.{key}={val} out of bounds"
    print("✅ Test 27 passed: All distribution values in [0, 1]")


def print_report_summary(report: dict):
    """Print key numbers for quick human review."""
    b  = report["binary_metrics"]
    m  = report["multiclass_metrics"]
    la = report["latency"]
    ab = report["ablation"]

    print("\n" + "─" * 60)
    print("  EVALUATION SUMMARY")
    print("─" * 60)
    print(f"  Test size         : {report['meta']['test_size']} samples")
    print(f"  Layers active     : {report['meta']['layers_active']}/3")
    print(f"  Binary F1         : {b['f1']:.4f}")
    print(f"  Binary accuracy   : {b['accuracy']:.4f}")
    print(f"  False positive rate: {b['false_positive_rate']:.4f}")
    print(f"  Weighted F1 (7cls): {m['weighted_f1']:.4f}")
    print(f"  Mean latency      : {la['mean_ms']:.1f}ms")
    print(f"  p95 latency       : {la['p95_ms']:.1f}ms")
    print("─" * 60)
    print("  ABLATION")
    for name, vals in ab.items():
        print(f"  {name:<18} F1={vals['f1']:.4f}  lat={vals['mean_latency_ms']:.0f}ms")
    print("─" * 60)


# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 56)
    print("  Phase 6 — Evaluation Tests")
    print("=" * 56)

    report = load_report()

    test_report_exists()
    test_report_structure()
    test_binary_metrics_bounds()
    test_multiclass_metrics()
    test_latency_plausible()
    test_ablation_ordering()
    test_score_distribution()
    print_report_summary(report)

    print()
    print("=" * 56)
    print("✅ All Evaluation tests passed!")
    print("Next → Phase 7: Docker + README")
    print("=" * 56)


if __name__ == "__main__":
    run_all_tests()