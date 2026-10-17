# tests/conftest.py
"""
Shared fixtures.

Most tests need no trained models and run everywhere (including CI).
Tests that use the `full_engine`, `classifier`, `matcher` or `report`
fixtures are skipped automatically when those files are missing.
"""

import importlib.util
import json
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LOG_TO_FILE", "0")

MODEL_DIR = os.path.join(ROOT, "models", "jailbreak_classifier", "best")
INDEX_DIR = os.path.join(ROOT, "models", "faiss_index")
REPORT_FILE = os.path.join(ROOT, "data", "results", "evaluation_report.json")


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def classifier_ready() -> bool:
    return installed("torch") and os.path.isdir(MODEL_DIR)


def index_ready() -> bool:
    return installed("faiss") and os.path.isfile(os.path.join(INDEX_DIR, "index.faiss"))


@pytest.fixture(scope="session")
def rule_only_engine():
    from src.jailbreak_detection.aggregator.risk_engine import RiskEngine
    return RiskEngine(load_classifier=False, load_embedding=False)


@pytest.fixture(scope="session")
def full_engine():
    if not (classifier_ready() and index_ready()):
        pytest.skip("trained classifier or FAISS index not found")
    from src.jailbreak_detection.aggregator.risk_engine import RiskEngine
    return RiskEngine(load_classifier=True, load_embedding=True)


@pytest.fixture(scope="session")
def classifier():
    if not classifier_ready():
        pytest.skip("trained classifier not found")
    from src.jailbreak_detection.detection.ml_classifier import JailbreakClassifier
    return JailbreakClassifier.load(MODEL_DIR)


@pytest.fixture(scope="session")
def matcher():
    if not index_ready():
        pytest.skip("FAISS index not found")
    from src.jailbreak_detection.detection.embedding_matcher import EmbeddingMatcher
    m = EmbeddingMatcher()
    assert m.load(INDEX_DIR)
    return m


@pytest.fixture(scope="session")
def report():
    if not os.path.isfile(REPORT_FILE):
        pytest.skip("no evaluation report; run scripts/evaluate.py")
    with open(REPORT_FILE, encoding="utf-8") as f:
        return json.load(f)


def make_client(engine):
    """API test client around a given engine (the app's model loading is skipped)."""
    from fastapi.testclient import TestClient

    from src.jailbreak_detection.api.main import create_app

    app = create_app()
    app.state.engine = engine
    app.state.stats = {"total_analyzed": 0, "total_blocked": 0, "total_monitored": 0, "total_allowed": 0}
    app.state.start_time = time.time()
    return TestClient(app)
