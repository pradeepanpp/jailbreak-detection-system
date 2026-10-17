# src/jailbreak_detection/detection/__init__.py
"""
Detection layers.

The ML classifier needs PyTorch, so it is imported lazily: importing the
rule engine (or running the model-free tests) does not load torch.
"""
from .rule_engine import RuleEngine, RuleEngineResult, RuleMatch

__all__ = [
    "RuleEngine",
    "RuleEngineResult",
    "RuleMatch",
    "JailbreakClassifier",
    "ClassifierResult",
]


def __getattr__(name):
    if name in ("JailbreakClassifier", "ClassifierResult"):
        from . import ml_classifier
        return getattr(ml_classifier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
