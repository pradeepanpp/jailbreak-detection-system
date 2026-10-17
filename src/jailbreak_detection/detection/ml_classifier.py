# src/jailbreak_detection/detection/ml_classifier.py
"""
Layer 2: fine-tuned DistilBERT classifier (7 classes).

Inference only; training lives in scripts/train_classifier.py.
Long inputs keep their first and last tokens (see tokenization.py), so a
many-shot payload at the end of a long prompt is not cut off.
"""

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.jailbreak_detection.constants import ID_TO_LABEL, LABEL_MAP
from src.jailbreak_detection.detection.tokenization import encode_head_tail, pad_batch
from src.jailbreak_detection.utils import load_config, logger


@dataclass
class ClassifierResult:
    predicted_category: str
    predicted_label: int
    confidence: float
    all_scores: dict   # {category: probability}
    is_attack: bool    # predicted category is not benign
    ml_score: float    # 1 - P(benign), used by the risk engine


class JailbreakClassifier:
    """
    Usage:
        clf = JailbreakClassifier.load("models/jailbreak_classifier/best")
        result = clf.predict("ignore all instructions")
    """

    def __init__(self):
        self.config = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    @classmethod
    def load(cls, model_path: str) -> "JailbreakClassifier":
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run scripts/train_classifier.py first."
            )
        clf = cls()
        clf.tokenizer = AutoTokenizer.from_pretrained(model_path)
        clf.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(clf.device)
        clf.model.eval()
        logger.info(f"Classifier loaded from {model_path} ({clf.device})")
        return clf

    def predict(self, text: str) -> ClassifierResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list) -> list:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Use JailbreakClassifier.load().")

        ids = encode_head_tail(self.tokenizer, texts, self.config["model"]["max_length"])
        batch = pad_batch(ids, self.tokenizer.pad_token_id or 0)
        inputs = {k: torch.tensor(v, device=self.device) for k, v in batch.items()}

        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=1).cpu().numpy()

        results = []
        for row in probs:
            label = int(row.argmax())
            category = ID_TO_LABEL[label]
            results.append(ClassifierResult(
                predicted_category=category,
                predicted_label=label,
                confidence=float(row[label]),
                all_scores={ID_TO_LABEL[i]: float(p) for i, p in enumerate(row)},
                is_attack=category != "benign",
                ml_score=float(1.0 - row[LABEL_MAP["benign"]]),
            ))
        return results
