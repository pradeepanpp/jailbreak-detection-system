# scripts/evaluate.py
"""
Phase 6 — Evaluation & Benchmarking.

Runs the complete detection pipeline against the held-out test set
and produces a comprehensive evaluation report.

Metrics computed:
  - Per-class precision, recall, F1
  - Macro + weighted F1, accuracy
  - Confusion matrix
  - Layer ablation (rule-only, rule+ml, all-three)
  - Latency benchmarks (p50, p95, p99)
  - False positive rate on benign subset
  - Risk score distribution per class

Outputs:
  data/results/evaluation_report.json   ← machine-readable full results
  data/results/evaluation_summary.txt   ← human-readable summary

Run:
    python scripts/evaluate.py
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
sys.path.append(".")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.jailbreak_detection.aggregator.risk_engine import RiskEngine
from src.jailbreak_detection.constants import (
    LABEL_NAMES,
    LABEL_MAP,
    ID_TO_LABEL,
    DECISION_BLOCK,
    DECISION_MONITOR,
    DECISION_ALLOW,
)
from src.jailbreak_detection.utils import logger, ensure_dirs


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TEST_CSV      = "data/processed/test.csv"
RESULTS_DIR   = "data/results"
REPORT_JSON   = f"{RESULTS_DIR}/evaluation_report.json"
REPORT_TXT    = f"{RESULTS_DIR}/evaluation_summary.txt"

# Attack categories (non-benign)
ATTACK_LABELS = [l for l in LABEL_NAMES if l != "benign"]

# Binary mapping: any non-benign → attack
BENIGN_ID     = LABEL_MAP["benign"]


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_test_data() -> pd.DataFrame:
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"Missing: {TEST_CSV}\n"
            "Run scripts/download_datasets.py first."
        )
    df = pd.read_csv(TEST_CSV)
    logger.info(f"Test set loaded: {len(df)} samples")
    return df


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def run_inference(engine: RiskEngine, texts: list) -> tuple:
    """
    Run engine on all test texts.
    Returns (results, latencies_ms).
    """
    results    = []
    latencies  = []

    for i, text in enumerate(texts):
        t0     = time.perf_counter()
        result = engine.analyze(text)
        t1     = time.perf_counter()

        latencies.append((t1 - t0) * 1000)   # ms
        results.append(result)

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{len(texts)} samples...")

    return results, latencies


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_binary_metrics(
    y_true_labels: list,
    results: list
) -> dict:
    """
    Binary classification: benign vs attack.
    Predicted attack = any decision != ALLOW.
    """
    y_true = [0 if l == "benign" else 1 for l in y_true_labels]
    y_pred = [0 if r.decision == DECISION_ALLOW else 1 for r in results]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(y_true)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # false positive rate

    return {
        "accuracy":          round(accuracy,  4),
        "precision":         round(precision, 4),
        "recall":            round(recall,    4),
        "f1":                round(f1,        4),
        "false_positive_rate": round(fpr,     4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def compute_multiclass_metrics(
    y_true_labels: list,
    results: list
) -> dict:
    """
    7-class classification using attack_category from results.
    Only meaningful when all 3 layers are active.
    """
    y_true = y_true_labels
    y_pred = [r.attack_category for r in results]

    # Per-class report
    report = classification_report(
        y_true, y_pred,
        labels       = LABEL_NAMES,
        output_dict  = True,
        zero_division = 0,
    )

    weighted_f1 = f1_score(y_true, y_pred, average="weighted",
                           labels=LABEL_NAMES, zero_division=0)
    macro_f1    = f1_score(y_true, y_pred, average="macro",
                           labels=LABEL_NAMES, zero_division=0)
    accuracy    = accuracy_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)

    return {
        "weighted_f1": round(weighted_f1, 4),
        "macro_f1":    round(macro_f1,    4),
        "accuracy":    round(accuracy,    4),
        "per_class":   {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1":        round(report[cls]["f1-score"],  4),
                "support":   int(report[cls]["support"]),
            }
            for cls in LABEL_NAMES if cls in report
        },
        "confusion_matrix": {
            "labels": LABEL_NAMES,
            "matrix": cm.tolist(),
        }
    }


def compute_latency_stats(latencies: list) -> dict:
    """p50, p95, p99, mean, min, max latency in ms."""
    arr = np.array(latencies)
    return {
        "mean_ms":  round(float(np.mean(arr)),          2),
        "min_ms":   round(float(np.min(arr)),           2),
        "max_ms":   round(float(np.max(arr)),           2),
        "p50_ms":   round(float(np.percentile(arr, 50)), 2),
        "p95_ms":   round(float(np.percentile(arr, 95)), 2),
        "p99_ms":   round(float(np.percentile(arr, 99)), 2),
        "total_ms": round(float(np.sum(arr)),            2),
    }


def compute_score_distribution(
    y_true_labels: list,
    results: list
) -> dict:
    """Mean risk score per true label — useful for calibration analysis."""
    dist = {}
    for label in LABEL_NAMES:
        scores = [
            r.risk_score
            for r, true_l in zip(results, y_true_labels)
            if true_l == label
        ]
        if scores:
            dist[label] = {
                "mean":   round(float(np.mean(scores)),   4),
                "median": round(float(np.median(scores)), 4),
                "min":    round(float(np.min(scores)),    4),
                "max":    round(float(np.max(scores)),    4),
            }
    return dist


# ─────────────────────────────────────────────
# LAYER ABLATION
# ─────────────────────────────────────────────

def run_ablation(texts: list, y_true_labels: list) -> dict:
    """
    Compare binary F1 across three engine configurations:
      1. Rule-only
      2. Rule + ML (no embedding)
      3. All three layers (full system)
    """
    logger.info("Running layer ablation study...")

    configs = [
        ("rule_only",    RiskEngine(load_classifier=False, load_embedding=False)),
        ("rule_ml",      RiskEngine(load_classifier=True,  load_embedding=False)),
        ("full_system",  RiskEngine(load_classifier=True,  load_embedding=True)),
    ]

    ablation = {}
    for name, engine in configs:
        logger.info(f"  Ablation: {name}...")
        results, latencies = run_inference(engine, texts)
        binary  = compute_binary_metrics(y_true_labels, results)
        lat     = compute_latency_stats(latencies)
        ablation[name] = {
            "f1":       binary["f1"],
            "accuracy": binary["accuracy"],
            "recall":   binary["recall"],
            "fpr":      binary["false_positive_rate"],
            "mean_latency_ms": lat["mean_ms"],
        }
        logger.info(
            f"    F1={binary['f1']:.4f}  "
            f"acc={binary['accuracy']:.4f}  "
            f"FPR={binary['false_positive_rate']:.4f}  "
            f"lat={lat['mean_ms']:.1f}ms"
        )

    return ablation


# ─────────────────────────────────────────────
# REPORT FORMATTING
# ─────────────────────────────────────────────

def format_summary(report: dict) -> str:
    """Build human-readable text summary."""
    b  = report["binary_metrics"]
    m  = report["multiclass_metrics"]
    la = report["latency"]
    ab = report["ablation"]

    lines = [
        "=" * 60,
        "  JAILBREAK DETECTION — EVALUATION REPORT",
        "=" * 60,
        f"  Test set : {report['meta']['test_size']} samples",
        f"  Layers   : {report['meta']['layers_active']}/3 active",
        "",
        "── BINARY CLASSIFICATION (attack vs benign) ──",
        f"  Accuracy  : {b['accuracy']:.4f}",
        f"  Precision : {b['precision']:.4f}",
        f"  Recall    : {b['recall']:.4f}",
        f"  F1        : {b['f1']:.4f}",
        f"  FP Rate   : {b['false_positive_rate']:.4f}",
        f"  TP={b['tp']}  TN={b['tn']}  FP={b['fp']}  FN={b['fn']}",
        "",
        "── MULTI-CLASS (7-class) ──",
        f"  Weighted F1 : {m['weighted_f1']:.4f}",
        f"  Macro F1    : {m['macro_f1']:.4f}",
        f"  Accuracy    : {m['accuracy']:.4f}",
        "",
        "── PER-CLASS F1 ──",
    ]

    for cls in LABEL_NAMES:
        if cls in m["per_class"]:
            pc  = m["per_class"][cls]
            bar = "█" * int(pc["f1"] * 20)
            lines.append(
                f"  {cls:<22} F1={pc['f1']:.3f}  "
                f"P={pc['precision']:.3f}  R={pc['recall']:.3f}  "
                f"n={pc['support']}  {bar}"
            )

    lines += [
        "",
        "── LATENCY (per sample, CPU) ──",
        f"  Mean : {la['mean_ms']:.1f}ms",
        f"  p50  : {la['p50_ms']:.1f}ms",
        f"  p95  : {la['p95_ms']:.1f}ms",
        f"  p99  : {la['p99_ms']:.1f}ms",
        f"  Total: {la['total_ms']:.0f}ms for {report['meta']['test_size']} samples",
        "",
        "── LAYER ABLATION (binary F1) ──",
    ]

    for name, vals in ab.items():
        lines.append(
            f"  {name:<18} F1={vals['f1']:.4f}  "
            f"recall={vals['recall']:.4f}  "
            f"FPR={vals['fpr']:.4f}  "
            f"lat={vals['mean_latency_ms']:.1f}ms"
        )

    lines += [
        "",
        "── RISK SCORE DISTRIBUTION (mean per class) ──",
    ]
    for cls, stats in report["score_distribution"].items():
        lines.append(
            f"  {cls:<22} mean={stats['mean']:.3f}  "
            f"[{stats['min']:.3f} – {stats['max']:.3f}]"
        )

    lines += [
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  Phase 6 — Evaluation & Benchmarking")
    logger.info("=" * 60)

    ensure_dirs(RESULTS_DIR)

    # ── 1. Load test data ─────────────────────
    logger.info("\nLoading test data...")
    df     = load_test_data()
    texts  = df["text"].tolist()
    labels = df["label_name"].tolist()

    # Print class distribution
    logger.info("\nTest set distribution:")
    for cls in LABEL_NAMES:
        count = labels.count(cls)
        logger.info(f"  {cls:<22} {count}")

    # ── 2. Load full engine ───────────────────
    logger.info("\nLoading full RiskEngine (all 3 layers)...")
    engine = RiskEngine(load_classifier=True, load_embedding=True)
    status = engine.status()
    logger.info(f"Layers active: {status['layers_active']}/3")

    # ── 3. Run full inference ─────────────────
    logger.info(f"\nRunning inference on {len(texts)} test samples...")
    results, latencies = run_inference(engine, texts)

    # ── 4. Compute metrics ────────────────────
    logger.info("\nComputing metrics...")

    binary     = compute_binary_metrics(labels, results)
    multiclass = compute_multiclass_metrics(labels, results)
    latency    = compute_latency_stats(latencies)
    score_dist = compute_score_distribution(labels, results)

    # ── 5. Layer ablation ─────────────────────
    logger.info("\nRunning ablation study (3 configurations)...")
    ablation = run_ablation(texts, labels)

    # ── 6. Assemble report ────────────────────
    report = {
        "meta": {
            "test_size":     len(texts),
            "layers_active": status["layers_active"],
            "rule_engine":   status["rule_engine"],
            "ml_classifier": status["ml_classifier"],
            "embedding":     status["embedding"],
        },
        "binary_metrics":    binary,
        "multiclass_metrics": multiclass,
        "latency":           latency,
        "ablation":          ablation,
        "score_distribution": score_dist,
    }

    # ── 7. Save outputs ───────────────────────
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport JSON saved → {REPORT_JSON}")

    summary = format_summary(report)
    with open(REPORT_TXT, "w") as f:
        f.write(summary)
    logger.info(f"Summary TXT saved  → {REPORT_TXT}")

    # ── 8. Print summary ──────────────────────
    print("\n" + summary)

    logger.info("=" * 60)
    logger.info("  Evaluation complete.")
    logger.info("=" * 60)
    logger.info(f"Next → Phase 7: Docker + README")


if __name__ == "__main__":
    main()