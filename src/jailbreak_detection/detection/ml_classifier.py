# src/jailbreak_detection/detection/ml_classifier.py
"""
Layer 2 — ML Classifier.

Fine-tunes DeBERTa-v3-base on jailbreak datasets.
Multi-class classifier predicting attack category.

Input  : raw text string
Output : predicted category + confidence scores per class
"""

import os
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report
)

from src.jailbreak_detection.utils import logger, load_config
from src.jailbreak_detection.constants import LABEL_MAP, ID_TO_LABEL


# -----------------------------------------------
# DATA CLASS
# -----------------------------------------------

@dataclass
class ClassifierResult:
    """Result from ML classifier"""
    predicted_category: str
    predicted_label:    int
    confidence:         float
    all_scores:         dict   # {category: score} for all classes
    is_attack:          bool   # True if not benign
    ml_score:           float  # 1 - benign_score (used by aggregator)


# -----------------------------------------------
# CLASSIFIER
# -----------------------------------------------

class JailbreakClassifier:
    """
    Fine-tuned DeBERTa-v3-base for jailbreak detection.

    Usage:
        # Training
        clf = JailbreakClassifier()
        clf.train(train_df, val_df)

        # Inference
        clf = JailbreakClassifier.load("models/classifier")
        result = clf.predict("ignore all instructions")
    """

    MODEL_NAME = "microsoft/deberta-v3-base"
    SAVE_PATH  = "models/jailbreak_classifier"

    def __init__(self):
        self.config    = load_config()
        self.device    = self._get_device()
        self.tokenizer = None
        self.model     = None

        logger.info(f"JailbreakClassifier initialized on {self.device}")

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            logger.info("GPU detected — using CUDA")
            return torch.device("cuda")
        else:
            logger.info("No GPU — using CPU")
            return torch.device("cpu")

    # -----------------------------------------------
    # TRAINING
    # -----------------------------------------------

    def train(self, train_df, val_df):
        """
        Fine-tune DeBERTa-v3-base on your dataset.

        Args:
            train_df : DataFrame with columns [text, label]
            val_df   : DataFrame with columns [text, label]
        """
        logger.info("Loading tokenizer and model...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=len(LABEL_MAP)
        ).to(self.device)

        logger.info(f"Model loaded — {len(LABEL_MAP)} output classes")

        # Tokenize datasets
        train_dataset = self._tokenize_df(train_df)
        val_dataset   = self._tokenize_df(val_df)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.SAVE_PATH,
            num_train_epochs=self.config["model"]["num_epochs"],
            per_device_train_batch_size=self.config["model"]["batch_size"],
            per_device_eval_batch_size=self.config["model"]["batch_size"] * 2,
            learning_rate=self.config["model"]["learning_rate"],
            warmup_ratio=0.1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir="logs/trainer",
            logging_steps=50,
            fp16=torch.cuda.is_available(),  # Mixed precision if GPU
            report_to="none"                 # No wandb/tensorboard
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2)
            ]
        )

        logger.info("Starting training...")
        trainer.train()

        # Save best model and tokenizer
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        self.model.save_pretrained(f"{self.SAVE_PATH}/best")
        self.tokenizer.save_pretrained(f"{self.SAVE_PATH}/best")
        logger.info(f"Model saved → {self.SAVE_PATH}/best")

    def _tokenize_df(self, df) -> Dataset:
        """Convert DataFrame to HuggingFace Dataset and tokenize"""

        dataset = Dataset.from_pandas(df[["text", "label"]])

        def tokenize_fn(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config["model"]["max_length"]
            )

        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"]
        )
        tokenized = tokenized.rename_column("label", "labels")
        tokenized.set_format("torch")

        return tokenized

    def _compute_metrics(self, eval_pred) -> dict:
        """Metrics computed after each epoch during training"""
        logits, labels = eval_pred
        predictions    = np.argmax(logits, axis=-1)

        return {
            "f1": f1_score(
                labels, predictions,
                average="weighted",
                zero_division=0
            ),
            "accuracy": accuracy_score(labels, predictions)
        }

    # -----------------------------------------------
    # INFERENCE
    # -----------------------------------------------

    def predict(self, text: str) -> ClassifierResult:
        """
        Predict attack category for a single text input.

        Args:
            text: Input text to classify
                  (use text_for_detection from normalizer)

        Returns:
            ClassifierResult with category, confidence, all scores
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError(
                "Model not loaded. Call JailbreakClassifier.load() first."
            )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config["model"]["max_length"]
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=1)

        probs_np        = probs[0].cpu().numpy()
        predicted_label = int(np.argmax(probs_np))
        confidence      = float(probs_np[predicted_label])

        # Build all scores dict
        all_scores = {
            ID_TO_LABEL[i]: float(probs_np[i])
            for i in range(len(probs_np))
        }

        predicted_category = ID_TO_LABEL[predicted_label]

        return ClassifierResult(
            predicted_category=predicted_category,
            predicted_label=predicted_label,
            confidence=confidence,
            all_scores=all_scores,
            is_attack=(predicted_category != "benign"),
            ml_score=float(1.0 - probs_np[LABEL_MAP["benign"]])
        )

    def predict_batch(self, texts: list) -> list:
        """
        Predict for a list of texts.
        More efficient than calling predict() in a loop.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded.")

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config["model"]["max_length"]
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=1)

        results = []
        for i in range(len(texts)):
            probs_i         = probs[i].cpu().numpy()
            predicted_label = int(np.argmax(probs_i))
            confidence      = float(probs_i[predicted_label])
            predicted_cat   = ID_TO_LABEL[predicted_label]

            results.append(ClassifierResult(
                predicted_category=predicted_cat,
                predicted_label=predicted_label,
                confidence=confidence,
                all_scores={
                    ID_TO_LABEL[j]: float(probs_i[j])
                    for j in range(len(probs_i))
                },
                is_attack=(predicted_cat != "benign"),
                ml_score=float(1.0 - probs_i[LABEL_MAP["benign"]])
            ))

        return results

    # -----------------------------------------------
    # SAVE / LOAD
    # -----------------------------------------------

    @classmethod
    def load(cls, model_path: str) -> "JailbreakClassifier":
        """
        Load a trained model from disk.

        Args:
            model_path: Path to saved model directory

        Returns:
            Loaded JailbreakClassifier ready for inference
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                f"Run: python scripts/train_classifier.py first"
            )

        instance           = cls()
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.model     = (
            AutoModelForSequenceClassification
            .from_pretrained(model_path)
            .to(instance.device)
        )
        instance.model.eval()

        logger.info(f"Model loaded from {model_path}")
        return instance

    def evaluate(self, test_df) -> dict:
        """
        Full evaluation on test set.
        Prints classification report and returns metrics dict.
        """
        logger.info(f"Evaluating on {len(test_df)} samples...")

        texts  = test_df["text"].tolist()
        labels = test_df["label"].tolist()

        results     = self.predict_batch(texts)
        predictions = [r.predicted_label for r in results]

        report = classification_report(
            labels,
            predictions,
            target_names=list(LABEL_MAP.keys()),
            zero_division=0
        )

        f1  = f1_score(labels, predictions, average="weighted", zero_division=0)
        acc = accuracy_score(labels, predictions)

        logger.info(f"\n{report}")
        logger.info(f"Weighted F1 : {f1:.4f}")
        logger.info(f"Accuracy    : {acc:.4f}")

        return {
            "f1":       f1,
            "accuracy": acc,
            "report":   report
        }