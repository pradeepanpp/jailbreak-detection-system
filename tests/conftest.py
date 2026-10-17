# tests/conftest.py
"""
Shared test setup.

Two kinds of tests live here:
  * model-free  — test_offline.py, test_preprocessing.py, test_rule_engine.py
                  Run everywhere, including CI (no PyTorch, no trained models).
  * model-based — need the trained classifier / FAISS index / an evaluation
                  report. They are skipped automatically when those
                  artefacts are missing, instead of crashing at import time.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LOG_TO_FILE", "0")

MODEL_DIR   = os.path.join(ROOT, "models", "jailbreak_classifier", "best")
INDEX_FILE  = os.path.join(ROOT, "models", "faiss_index", "index.faiss")
REPORT_FILE = os.path.join(ROOT, "data", "results", "evaluation_report.json")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


_models_ready = _torch_available() and os.path.isdir(MODEL_DIR) and os.path.isfile(INDEX_FILE)

collect_ignore = []
if not _models_ready:
    collect_ignore += [
        "test_aggregator.py",
        "test_api.py",
        "test_classifier.py",
        "test_embedding_matcher.py",
    ]
if not os.path.isfile(REPORT_FILE):
    collect_ignore.append("test_evaluation.py")
