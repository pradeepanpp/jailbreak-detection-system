# scripts/train_classifier.py
"""
Fine-tune DistilBERT on data/processed/{train,val,test}.csv.

    python scripts/train_classifier.py                         # default weights
    python scripts/train_classifier.py --class-weights balanced

Saves the best checkpoint (lowest validation loss) to
models/jailbreak_classifier/best and prints a per-class test report.
"""

import argparse
import os
import sys
sys.path.append(".")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.jailbreak_detection.constants import ID_TO_LABEL, LABEL_MAP
from src.jailbreak_detection.detection.tokenization import encode_head_tail, pad_batch
from src.jailbreak_detection.utils import load_config, logger

MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "models/jailbreak_classifier"
DATA_DIR = "data/processed"
NUM_LABELS = len(LABEL_MAP)


class WeightedLossTrainer(Trainer):
    """Trainer that uses class-weighted cross-entropy."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(next(self.model.parameters()).device)
        # Our loss is a plain batch mean. Without this flag the Trainer
        # assumes it is already scaled for gradient accumulation and does
        # not divide it, so gradients come out too large.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_splits():
    for name in ["train.csv", "val.csv", "test.csv"]:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run scripts/download_datasets.py first.")

    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    logger.info(f"Train {len(train_df)}, val {len(val_df)}, test {len(test_df)} samples")
    return train_df, val_df, test_df


def compute_weights(train_df: pd.DataFrame, binary_balanced: bool = True) -> torch.Tensor:
    """
    Inverse-frequency class weights.

    Plain "balanced" weights give each of the 7 classes the same share of
    the loss, so the 6 attack classes together outweigh benign 6 to 1 and
    borderline inputs get pushed toward "attack". With binary_balanced the
    benign weight is multiplied by 6, so benign and attacks weigh the same
    in total while the attack classes stay balanced against each other.
    """
    labels = train_df["label"].values
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(NUM_LABELS), y=labels)
    if binary_balanced:
        weights[LABEL_MAP["benign"]] *= NUM_LABELS - 1

    logger.info(f"Class weights ({'binary-balanced' if binary_balanced else 'per-class balanced'}):")
    for i, w in enumerate(weights):
        logger.info(f"  {ID_TO_LABEL[i]:<20} count={(labels == i).sum():>3}  weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32)


def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tokenize(batch):
        ids = encode_head_tail(tokenizer, batch["text"], max_length)
        return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenizing")
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return dataset


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
    }


def get_training_args(config: dict, device: torch.device, n_train: int) -> TrainingArguments:
    """
    Batch 16 with no gradient accumulation gives 18 updates per epoch on
    ~275 samples. Warm-up is one epoch. The best checkpoint is picked by
    validation loss, because F1 on 61 validation samples stays flat in
    early epochs and stopped training too soon.
    """
    on_gpu = device.type == "cuda"
    batch_size = config["model"]["batch_size"]
    steps_per_epoch = -(-n_train // batch_size)  # ceiling division

    return TrainingArguments(
        output_dir=SAVE_PATH,
        num_train_epochs=config["model"]["num_epochs"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=float(config["model"]["learning_rate"]),
        warmup_steps=steps_per_epoch,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        fp16=on_gpu,
        dataloader_num_workers=0,  # Windows-safe
        dataloader_pin_memory=on_gpu,
        report_to="none",
        save_total_limit=2,
    )


def evaluate_on_test(model, tokenizer, test_df: pd.DataFrame, max_length: int, device) -> dict:
    """Classifier alone on the test split: per-class report and confusion matrix."""
    model.eval()
    model.to(device)

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()
    preds = []
    for i in range(0, len(texts), 16):
        ids = encode_head_tail(tokenizer, texts[i:i + 16], max_length)
        batch = pad_batch(ids, tokenizer.pad_token_id or 0)
        inputs = {k: torch.tensor(v, device=device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds += torch.argmax(logits, dim=1).cpu().tolist()

    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=list(LABEL_MAP.keys()),
                                   zero_division=0, digits=4)
    logger.info(f"Test set (classifier only):\n{report}")

    logger.info("Confusion matrix (rows = actual, columns = predicted):")
    logger.info("  " + " ".join(f"{ID_TO_LABEL[i][:6]:>7}" for i in range(NUM_LABELS)))
    for i, row in enumerate(confusion_matrix(labels, preds)):
        logger.info(f"  {ID_TO_LABEL[i][:6]:<8}" + " ".join(f"{v:>7}" for v in row))

    logger.info(f"Weighted F1 {f1:.4f}, accuracy {acc:.4f}")
    return {"f1": f1, "accuracy": acc}


def save_results(metrics: dict, train_size: int):
    os.makedirs("data/results", exist_ok=True)
    row = {
        "model": MODEL_NAME,
        "train_size": train_size,
        "f1_weighted": round(metrics["f1"], 4),
        "accuracy": round(metrics["accuracy"], 4),
    }
    pd.DataFrame([row]).to_csv("data/results/results.csv", index=False)


def main(binary_balanced: bool = True):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = config["model"]["max_length"]
    logger.info(f"Training {MODEL_NAME} on {device}")

    train_df, val_df, test_df = load_splits()
    class_weights = compute_weights(train_df, binary_balanced)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)

    train_dataset = tokenize_dataset(train_df, tokenizer, max_length)
    val_dataset = tokenize_dataset(val_df, tokenizer, max_length)
    training_args = get_training_args(config, device, len(train_df))
    logger.info(
        f"Batch {training_args.per_device_train_batch_size}, "
        f"learning rate {training_args.learning_rate}, up to {training_args.num_train_epochs} epochs"
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    result = trainer.train()
    logger.info(f"Training finished after {result.global_step} steps (loss {result.training_loss:.4f})")

    best_path = f"{SAVE_PATH}/best"
    os.makedirs(best_path, exist_ok=True)
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    logger.info(f"Best model saved to {best_path}")

    metrics = evaluate_on_test(trainer.model, tokenizer, test_df, max_length, device)
    save_results(metrics, len(train_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the jailbreak classifier")
    parser.add_argument(
        "--class-weights", choices=["binary", "balanced"], default="binary",
        help="binary: benign weighs as much as all attacks together (default); "
             "balanced: plain per-class inverse frequency",
    )
    args = parser.parse_args()
    main(binary_balanced=args.class_weights == "binary")
