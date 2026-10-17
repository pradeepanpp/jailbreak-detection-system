# scripts/evaluate.py
"""
Phase 6 — Evaluation & Benchmarking.

Runs the complete detection pipeline against the held-out test set
and produces a comprehensive evaluation report.

Evaluation sets:
  test.csv            in-distribution test split (group-aware, no near-dups)
  hard_negatives.csv  benign prompts that look suspicious, never trained on
                      → the realistic false-positive rate
  external_test.csv   WildGuard test set, never trained on (if available)
                      → generalisation to prompts written by other people

All rates come with 95% Wilson confidence intervals: with ~50 test
samples, one error moves recall by ~3 points, so intervals matter.

Metrics computed:
  - Per-class precision, recall, F1
  - Macro + weighted F1, accuracy
  - Confusion matrix
  - Layer ablation (rule-only, rule+ml, all-three)
  - Latency benchmarks (p50, p95, p99)
  - False positive rate on benign subset and on held-out hard negatives
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
    accuracy_score,
)

from src.jailbreak_detection.aggregator.risk_engine import RiskEngine
from src.jailbreak_detection.constants import (
    LABEL_NAMES,
    LABEL_MAP,
    DECISION_BLOCK,
    DECISION_ALLOW,
)
from src.jailbreak_detection.utils import logger, ensure_dirs


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TEST_CSV      = "data/processed/test.csv"
HARD_NEG_CSV  = "data/processed/hard_negatives.csv"
EXTERNAL_CSV  = "data/processed/external_test.csv"
RESULTS_DIR   = "data/results"
REPORT_JSON   = f"{RESULTS_DIR}/evaluation_report.json"
REPORT_TXT    = f"{RESULTS_DIR}/evaluation_summary.txt"

# Attack categories (non-benign)
ATTACK_LABELS = [name for name in LABEL_NAMES if name != "benign"]

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

def wilson_interval(k: int, n: int, z: float = 1.96) -> list:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return [0.0, 0.0]
    p      = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def compute_binary_metrics(
    y_true_labels: list,
    results: list
) -> dict:
    """
    Binary classification: benign vs attack.
    Predicted attack = any decision != ALLOW (MONITOR counts as flagged).
    """
    y_true = [0 if label == "benign" else 1 for label in y_true_labels]
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
        "recall_ci95":       wilson_interval(tp, tp + fn),
        "fpr_ci95":          wilson_interval(fp, fp + tn),
        "block_only_recall": round(
            sum(1 for t, r in zip(y_true, results) if t == 1 and r.decision == DECISION_BLOCK)
            / max(1, tp + fn), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def evaluate_extra_set(engine: RiskEngine, path: str) -> dict | None:
    """
    Binary metrics on an extra evaluation CSV (hard negatives / external).
    Returns None if the file does not exist.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    results, _ = run_inference(engine, df["text"].tolist())
    metrics = compute_binary_metrics(df["label_name"].tolist(), results)
    metrics["n"] = len(df)
    metrics["false_positives"] = [
        " ".join(t.split())[:120] for t, label, r in zip(df["text"], df["label_name"], results)
        if label == "benign" and r.decision != DECISION_ALLOW
    ]
    return metrics


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
    Compare binary F1 (test set) and hard-negative FPR across three
    engine configurations:
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
        hard    = evaluate_extra_set(engine, HARD_NEG_CSV)
        ablation[name] = {
            "f1":       binary["f1"],
            "accuracy": binary["accuracy"],
            "recall":   binary["recall"],
            "fpr":      binary["false_positive_rate"],
            "hard_negative_fpr": hard["false_positive_rate"] if hard else None,
            "mean_latency_ms": lat["mean_ms"],
            "p50_latency_ms":  lat["p50_ms"],
        }
        logger.info(
            f"    F1={binary['f1']:.4f}  "
            f"acc={binary['accuracy']:.4f}  "
            f"FPR={binary['false_positive_rate']:.4f}  "
            f"hardFPR={ablation[name]['hard_negative_fpr']}  "
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
        f"  F1        : {b['f1']:.4f}",
        f"  Recall    : {b['recall']:.4f}  95% CI {b['recall_ci95']}",
        f"  FP Rate   : {b['false_positive_rate']:.4f}  95% CI {b['fpr_ci95']}",
        f"  TP={b['tp']}  TN={b['tn']}  FP={b['fp']}  FN={b['fn']}",
        "",
    ]

    hn = report.get("hard_negatives")
    if hn:
        lines += [
            "── HELD-OUT HARD NEGATIVES (benign, never trained on) ──",
            f"  FP Rate   : {hn['false_positive_rate']:.4f}  95% CI {hn['fpr_ci95']}  (n={hn['n']})",
        ] + [f"    FP: {t}" for t in hn["false_positives"]] + [""]

    ex = report.get("external_test")
    if ex:
        lines += [
            "── EXTERNAL TEST (WildGuard test, never trained on) ──",
            f"  F1        : {ex['f1']:.4f}   (n={ex['n']})",
            f"  Recall    : {ex['recall']:.4f}  95% CI {ex['recall_ci95']}",
            f"  FP Rate   : {ex['false_positive_rate']:.4f}  95% CI {ex['fpr_ci95']}",
            "",
        ]

    lines += [
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
            f"hard-neg FPR={vals['hard_negative_fpr']}  "
            f"p50={vals['p50_latency_ms']:.1f}ms"
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
    # Windows consoles (cp1252) cannot print Arabic/Malayalam test prompts
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

    logger.info("\nEvaluating held-out hard negatives and external test set...")
    hard_neg = evaluate_extra_set(engine, HARD_NEG_CSV)
    external = evaluate_extra_set(engine, EXTERNAL_CSV)

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
        "hard_negatives":    hard_neg,
        "external_test":     external,
        "multiclass_metrics": multiclass,
        "latency":           latency,
        "ablation":          ablation,
        "score_distribution": score_dist,
    }

    # ── 7. Save outputs ───────────────────────
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"\nReport JSON saved → {REPORT_JSON}")

    summary = format_summary(report)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Summary TXT saved  → {REPORT_TXT}")

    # ── 8. Print summary ──────────────────────
    print("\n" + summary)

    logger.info("=" * 60)
    logger.info("  Evaluation complete.")
    logger.info("=" * 60)
    logger.info("Send data/results/evaluation_summary.txt to update the README numbers.")


if __name__ == "__main__":
    main()