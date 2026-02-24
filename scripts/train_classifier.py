# scripts/train_classifier.py
"""
Fine-tunes DeBERTa-v3-base on the jailbreak dataset.
Run after download_datasets.py completes.

Usage: python scripts/train_classifier.py
"""

import sys
sys.path.append(".")

import pandas as pd
from src.jailbreak_detection.detection.ml_classifier import JailbreakClassifier
from src.jailbreak_detection.utils import logger


def main():
    logger.info("Loading processed datasets...")

    train_df = pd.read_csv("data/processed/train.csv")
    val_df   = pd.read_csv("data/processed/val.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    logger.info(f"Train : {len(train_df)} samples")
    logger.info(f"Val   : {len(val_df)} samples")
    logger.info(f"Test  : {len(test_df)} samples")

    # Train
    clf = JailbreakClassifier()
    clf.train(train_df, val_df)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    metrics = clf.evaluate(test_df)

    logger.info(f"\n✅ Training complete!")
    logger.info(f"   F1 Score : {metrics['f1']:.4f}")
    logger.info(f"   Accuracy : {metrics['accuracy']:.4f}")
    logger.info("Next → python scripts/build_index.py")


if __name__ == "__main__":
    main()