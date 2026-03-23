# scripts/train_classifier.py
"""
Phase 2 — Step 3: Fine-tune DeBERTa-v3-base on jailbreak dataset.

Key features:
  - Weighted cross-entropy loss (fixes class imbalance)
  - CPU-safe training arguments
  - Early stopping to prevent overfitting on small dataset
  - Full per-class evaluation report
  - Auto-saves best model checkpoint

Run after download_datasets.py:
  python scripts/train_classifier.py

Requirements:
  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv
"""

import sys
import os
sys.path.append(".")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from src.jailbreak_detection.utils import logger, load_config
from src.jailbreak_detection.constants import LABEL_MAP, ID_TO_LABEL


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH  = "models/jailbreak_classifier"
DATA_DIR   = "data/processed"
NUM_LABELS = len(LABEL_MAP)


# ─────────────────────────────────────────────
# WEIGHTED LOSS TRAINER
# Fixes the direct_jailbreak imbalance (120 vs ~35 for others)
# ─────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that applies per-class weights to the loss.

    Without this, the model learns to over-predict direct_jailbreak
    because it has 3x more samples than other categories.

    class_weights tensor is injected at instantiation.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(
            next(self.model.parameters()).device
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss    = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train / val / test CSVs produced by download_datasets.py"""
    required = ["train.csv", "val.csv", "test.csv"]
    for fname in required:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"Run: python scripts/download_datasets.py first"
            )

    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    val_df   = pd.read_csv(f"{DATA_DIR}/val.csv")
    test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")

    logger.info(f"Train : {len(train_df)} samples")
    logger.info(f"Val   : {len(val_df)} samples")
    logger.info(f"Test  : {len(test_df)} samples")

    return train_df, val_df, test_df


def compute_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training labels.

    Example for our dataset:
      direct_jailbreak = 84 samples → low weight
      encoding_attack  = 22 samples → high weight

    sklearn's compute_class_weight handles this automatically.
    """
    labels  = train_df["label"].values
    classes = np.arange(NUM_LABELS)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    logger.info("Class weights computed:")
    for i, w in enumerate(weights):
        label_name = ID_TO_LABEL[i]
        count      = (labels == i).sum()
        logger.info(f"  [{i}] {label_name:<22} count={count:>3}  weight={w:.4f}")

    return weight_tensor


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int
) -> Dataset:
    """Convert DataFrame → HuggingFace Dataset → tokenize"""

    dataset = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,      # DataCollatorWithPadding handles this
            max_length=max_length
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    return tokenized


def compute_metrics(eval_pred) -> dict:
    """Called after each epoch. Returns F1 and accuracy for Trainer."""
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


def print_split_distribution(df: pd.DataFrame, name: str):
    """Visual distribution check before training."""
    logger.info(f"\n  {name} distribution:")
    for label_id, label_name in ID_TO_LABEL.items():
        count = (df["label"] == label_id).sum()
        bar   = "█" * max(1, count // 3)
        logger.info(f"    [{label_id}] {label_name:<22} {count:>4}  {bar}")


def get_training_args(config: dict, device: torch.device) -> TrainingArguments:
    """
    Build TrainingArguments.
    CPU-safe: no fp16, smaller batch, gradient accumulation to compensate.
    """
    on_gpu = (device.type == "cuda")

    # On CPU with 233 training samples:
    # batch_size=8, grad_accum=4 → effective batch=32
    # This is stable and won't OOM on CPU
    batch_size = config["model"]["batch_size"] if on_gpu else 8
    grad_accum = 1 if on_gpu else 4

    return TrainingArguments(
        output_dir                  = SAVE_PATH,
        num_train_epochs            = config["model"]["num_epochs"],
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size * 2,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = float(config["model"]["learning_rate"]),
        warmup_ratio                = 0.1,
        weight_decay                = 0.01,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        logging_dir                 = "logs/trainer",
        logging_steps               = 10,
        fp16                        = on_gpu,       # CPU does not support fp16
        dataloader_num_workers      = 0,            # Windows-safe
        report_to                   = "none",
        save_total_limit            = 2,            # Keep only last 2 checkpoints
    )


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_on_test(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    config: dict,
    device: torch.device
):
    """
    Full evaluation on held-out test set.
    Prints per-class precision, recall, F1 and confusion matrix.
    """
    logger.info("\n" + "─" * 52)
    logger.info("  Final Evaluation on Test Set")
    logger.info("─" * 52)

    model.eval()
    model.to(device)

    all_preds  = []
    all_labels = []

    # Process in batches to avoid OOM
    batch_size = 16
    texts      = test_df["text"].tolist()
    labels     = test_df["label"].tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs      = tokenizer(
            batch_texts,
            return_tensors  = "pt",
            truncation      = True,
            padding         = True,
            max_length      = config["model"]["max_length"]
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels[i:i + batch_size])

    # Metrics
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        target_names = list(LABEL_MAP.keys()),
        zero_division = 0,
        digits        = 4
    )

    logger.info(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info("Confusion Matrix (rows=actual, cols=predicted):")
    header = "  " + " ".join(f"{ID_TO_LABEL[i][:6]:>7}" for i in range(NUM_LABELS))
    logger.info(header)
    for i, row in enumerate(cm):
        row_str = f"  {ID_TO_LABEL[i][:6]:<8}" + " ".join(f"{v:>7}" for v in row)
        logger.info(row_str)

    logger.info(f"\n  Weighted F1 : {f1:.4f}")
    logger.info(f"  Accuracy    : {acc:.4f}")

    return {"f1": f1, "accuracy": acc, "report": report}


def save_results(metrics: dict, train_df: pd.DataFrame):
    """Save evaluation results to data/results/results.csv"""
    os.makedirs("data/results", exist_ok=True)
    results = {
        "model":       MODEL_NAME,
        "train_size":  len(train_df),
        "f1_weighted": round(metrics["f1"], 4),
        "accuracy":    round(metrics["accuracy"], 4),
    }
    pd.DataFrame([results]).to_csv("data/results/results.csv", index=False)
    logger.info("Results saved → data/results/results.csv")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logger.info("=" * 52)
    logger.info("  Phase 2 — Step 3: Train Classifier")
    logger.info("=" * 52)

    # ── 1. Config & device ────────────────────────────
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    logger.info(f"Model  : {MODEL_NAME}")

    # ── 2. Load data ──────────────────────────────────
    logger.info("\nLoading datasets...")
    train_df, val_df, test_df = load_splits()

    # Distribution check
    print_split_distribution(train_df, "Train")
    print_split_distribution(val_df,   "Val")

    # ── 3. Class weights ──────────────────────────────
    logger.info("\nComputing class weights (fixing imbalance)...")
    class_weights = compute_weights(train_df)

    # ── 4. Load tokenizer & model ─────────────────────
    logger.info(f"\nLoading {MODEL_NAME}...")
    logger.info("(First run downloads ~180MB — this is normal)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_LABELS,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 5. Tokenize ───────────────────────────────────
    logger.info("\nTokenizing datasets...")
    max_length    = config["model"]["max_length"]
    train_dataset = tokenize_dataset(train_df, tokenizer, max_length)
    val_dataset   = tokenize_dataset(val_df,   tokenizer, max_length)

    logger.info(f"Train dataset: {len(train_dataset)} samples tokenized")
    logger.info(f"Val dataset  : {len(val_dataset)} samples tokenized")

    # ── 6. Training arguments ─────────────────────────
    training_args = get_training_args(config, device)

    logger.info(f"\nTraining config:")
    logger.info(f"  Epochs         : {training_args.num_train_epochs}")
    logger.info(f"  Batch size     : {training_args.per_device_train_batch_size}")
    logger.info(f"  Grad accum     : {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate  : {training_args.learning_rate}")
    logger.info(f"  FP16           : {training_args.fp16}")

    # ── 7. Trainer with weighted loss ─────────────────
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedLossTrainer(
        class_weights = class_weights,
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        compute_metrics = compute_metrics,
        data_collator = data_collator,
        callbacks     = [
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    # ── 8. Train ──────────────────────────────────────
    logger.info("\n" + "=" * 52)
    logger.info("  Starting Training")
    logger.info("=" * 52)
    logger.info("NOTE: On CPU this will take 10–30 min depending on your machine.")
    logger.info("Each epoch trains on 233 samples — watch F1 improve per epoch.\n")

    train_result = trainer.train()

    logger.info(f"\nTraining complete!")
    logger.info(f"  Total steps   : {train_result.global_step}")
    logger.info(f"  Training loss : {train_result.training_loss:.4f}")

    # ── 9. Save best model ────────────────────────────
    best_path = f"{SAVE_PATH}/best"
    os.makedirs(best_path, exist_ok=True)
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    logger.info(f"\nBest model saved → {best_path}")

    # ── 10. Test evaluation ───────────────────────────
    metrics = evaluate_on_test(
        model     = trainer.model,
        tokenizer = tokenizer,
        test_df   = test_df,
        config    = config,
        device    = device
    )

    # ── 11. Save results ──────────────────────────────
    save_results(metrics, train_df)

    # ── 12. Final summary ─────────────────────────────
    logger.info("\n" + "=" * 52)
    logger.info("  Training Pipeline Complete")
    logger.info("=" * 52)
    logger.info(f"  Model saved    : {best_path}")
    logger.info(f"  Test F1        : {metrics['f1']:.4f}")
    logger.info(f"  Test Accuracy  : {metrics['accuracy']:.4f}")

    # Interpret results for the user
    f1 = metrics["f1"]
    if f1 >= 0.85:
        quality = "Excellent — production ready"
    elif f1 >= 0.70:
        quality = "Good — solid for portfolio demo"
    elif f1 >= 0.55:
        quality = "Acceptable — add WildGuard data to improve"
    else:
        quality = "Low — WildGuard data will significantly help"

    logger.info(f"  Quality        : {quality}")
    logger.info("=" * 52)
    logger.info("Next → python scripts/build_index.py")


if __name__ == "__main__":
    main()