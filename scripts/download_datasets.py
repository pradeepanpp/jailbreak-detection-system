# scripts/download_datasets.py
"""
Build the train / val / test CSVs for the 7-class classifier.

The detector targets jailbreak TECHNIQUES (the wrapper), not harmful
topics, so plain harmful requests without a technique are left out.

Sources:
  WildGuard (gated, optional)  adversarial+harmful -> attack, unharmful -> benign;
                               its test split is saved as external_test.csv
  JailbreakBench               benign "Goal" prompts -> benign
  synthetic_data.py            hand-written attacks for all 6 attack classes,
                               plus benign hard negatives

The split is group-aware: near-duplicates (same many-shot template, same
base64 payload) always stay in one split, so the test set cannot be
answered by memorising training examples.

Usage:
  python scripts/download_datasets.py [--hf-token hf_xxx]
"""

import sys
import os
import argparse
import re
sys.path.append(".")

import pandas as pd
import numpy as np

import synthetic_data as sd
from src.jailbreak_detection.constants import LABEL_MAP, ID_TO_LABEL
from src.jailbreak_detection.utils import logger

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
RANDOM_SEED   = 42
MIN_SAMPLES   = 30   # Minimum per class to train reliably

# WildGuard has ~87k prompts. CPU training on all of them takes hours,
# so each bucket (vanilla benign / adversarial benign / adversarial attack)
# is capped. Raise with --wildguard-max if you have a GPU.
WILDGUARD_MAX_PER_BUCKET = 500


def _load_dataset(*args, **kwargs):
    """Import `datasets` lazily so the rest of the script works without it."""
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


def normalize_key(text: str) -> str:
    """Lowercase + collapse whitespace. Used for dedup and default groups."""
    return re.sub(r"\s+", " ", text.strip().lower())


def make_record(text: str, label_name: str, source: str, group: str = None) -> dict:
    """
    group: records sharing a group are near-duplicates and must stay in
    the same split. Defaults to the normalised text (each text its own group).
    """
    text = text.strip()
    return {
        "text":       text,
        "label":      LABEL_MAP[label_name],
        "label_name": label_name,
        "source":     source,
        "group":      group or normalize_key(text),
    }


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates by normalised text (so 'Human  trafficking' and
    'Human trafficking' count as one), and drop texts that appear with
    conflicting labels — they would teach the model contradictions.
    """
    before = len(df)
    key    = df["text"].map(normalize_key)

    n_labels  = df.groupby(key)["label"].transform("nunique")
    conflicts = int((n_labels > 1).sum())
    df        = df[n_labels == 1]
    key       = key[n_labels == 1]

    df = df[~key.duplicated()]
    logger.info(
        f"Deduplication: {before} -> {len(df)} "
        f"({conflicts} with conflicting labels dropped)"
    )
    return df.reset_index(drop=True)


def is_encoding_attack(text: str) -> bool:
    b64 = r'[A-Za-z0-9+/]{20,}={0,2}'
    if len(re.findall(b64, text)) >= 2:
        return True
    leet = re.findall(r'\b[a-z]*[0-9][a-z0-9]*\b', text, re.IGNORECASE)
    return len(leet) >= 4


def is_roleplay(text: str) -> bool:
    patterns = [
        r'\bwrite a (story|scene|script|novel|fiction)\b',
        r'\bpretend (you are|to be|that)\b',
        r'\bact as (a|an|if)\b',
        r'\brole.?play\b',
        r'\bimagine (you are|a world where)\b',
        r'\bin a fictional\b',
        r'\bas a character\b',
    ]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def is_prompt_injection(text: str) -> bool:
    patterns = [
        r'<s>', r'\[system\]',
        r'forget (all |your |previous )(instructions|rules)',
        r'new instructions?:',
        r'updated instructions?:',
        r'ignore (all |previous |the )(instructions|rules|above)',
    ]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def is_manyshot(text: str) -> bool:
    if len(text) < 1500:
        return False
    turn_patterns = [
        r'(human|user)\s*:\s*.+\n.*(assistant|ai)\s*:\s*',
        r'(q\s*:|question\s*:).+\n.*(a\s*:|answer\s*:)',
        r'example \d+',
    ]
    t = text.lower()
    return any(re.search(p, t, re.DOTALL) for p in turn_patterns)


def is_indirect_injection(text: str) -> bool:
    patterns = [
        r'<!--.*?(ignore|instructions|override).*?-->',
        r'/\*.*?(ignore|instructions|override).*?\*/',
        r'\[hidden\]', r'<!-- ?system',
    ]
    return any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in patterns)


def smart_label(text: str, fallback: str = "direct_jailbreak") -> str:
    """Regex-based category for external attack prompts (first match wins)."""
    checks = [
        (is_encoding_attack,    "encoding_attack"),
        (is_indirect_injection, "indirect_injection"),
        (is_manyshot,           "manyshot"),
        (is_prompt_injection,   "prompt_injection"),
        (is_roleplay,           "roleplay_hijack"),
    ]
    for check, label in checks:
        if check(text):
            return label
    return fallback


def _wildguard_record(row: dict, source: str) -> dict | None:
    """
    Map one WildGuardMix row to our schema, or None if out of scope.

    WildGuard labels are prompt_harm_label in {"harmful", "unharmful"}
    (never "benign") plus an `adversarial` flag that marks jailbreak-style
    prompts. Only adversarial + harmful prompts are jailbreak attacks.
    """
    text        = (row.get("prompt") or "").strip()
    harm        = (row.get("prompt_harm_label") or "").lower()
    adversarial = bool(row.get("adversarial"))
    if not text or harm not in ("harmful", "unharmful"):
        return None
    if harm == "unharmful":
        return make_record(text, "benign", source)
    if adversarial:
        return make_record(text, smart_label(text), source)
    return None   # plain harmful request, no jailbreak technique → out of scope


def load_wildguard(hf_token: str = None, max_per_bucket: int = WILDGUARD_MAX_PER_BUCKET) -> pd.DataFrame:
    """allenai/wildguardmix train split — gated, requires an HF token."""
    logger.info("Loading WildGuard (allenai/wildguardmix)...")
    records = []
    try:
        kwargs = {"token": hf_token} if hf_token else {}
        ds = _load_dataset("allenai/wildguardmix", "wildguardtrain", split="train", **kwargs)
        for row in ds:
            rec = _wildguard_record(row, "wildguard")
            if rec:
                rec["bucket"] = (
                    "attack" if rec["label_name"] != "benign"
                    else "benign_adversarial" if row.get("adversarial") else "benign_vanilla"
                )
                records.append(rec)

        df = pd.DataFrame(records)
        if len(df):
            df = (
                df.groupby("bucket", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), max_per_bucket), random_state=RANDOM_SEED))
                  .drop(columns="bucket")
                  .reset_index(drop=True)
            )
        logger.info(f"  WildGuard: {len(df)} samples (max {max_per_bucket} per bucket)")
        return df
    except Exception as e:
        logger.warning(f"  WildGuard failed: {e}")
        if "gated" in str(e).lower() or "authenticated" in str(e).lower():
            logger.warning(
                "  WildGuard is gated. To use it:\n"
                "  1. Accept the terms: https://huggingface.co/datasets/allenai/wildguardmix\n"
                "  2. Create a token:   https://huggingface.co/settings/tokens\n"
                "  3. Run: python scripts/download_datasets.py --hf-token hf_yourtoken"
            )
        return pd.DataFrame()


def load_wildguard_test(hf_token: str = None) -> pd.DataFrame:
    """
    allenai/wildguardmix 'wildguardtest' — an external test set written by
    different people than our training data. Saved separately and NEVER
    trained on, so it measures real generalisation.
    """
    try:
        kwargs = {"token": hf_token} if hf_token else {}
        ds = _load_dataset("allenai/wildguardmix", "wildguardtest", split="test", **kwargs)
        records = [r for r in (_wildguard_record(row, "wildguard_test") for row in ds) if r]
        logger.info(f"  WildGuard test (external): {len(records)} samples")
        return pd.DataFrame(records)
    except Exception as e:
        logger.warning(f"  WildGuard test set unavailable: {e}")
        return pd.DataFrame()


def load_jailbreakbench(hf_token: str = None) -> pd.DataFrame:
    """
    JailbreakBench/JBB-Behaviors — benign split only.

    Each row has a short 'Behavior' name ("Defamation") and a full
    'Goal' prompt ("Write a defamatory article claiming ..."). We use the
    Goal: the Behavior name is a topic label, not a prompt.

    Benign goals are sensitive-topic prompts with benign intent — good
    hard negatives. Harmful goals are plain harmful requests without any
    jailbreak technique, which is out of scope for this detector.
    """
    logger.info("Loading JailbreakBench (JailbreakBench/JBB-Behaviors, benign split)...")
    records = []
    try:
        kwargs = {"token": hf_token} if hf_token else {}
        ds_benign = _load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign", **kwargs)
        for row in ds_benign:
            text     = (row.get("Goal") or row.get("goal") or "").strip()
            behavior = (row.get("Behavior") or row.get("behavior") or "").strip()
            if not text:
                continue
            records.append(make_record(
                text, "benign", "jailbreakbench",
                group=f"jbb:{normalize_key(behavior or text)}",
            ))
        logger.info(f"  JailbreakBench: {len(records)} benign goal prompts")
    except Exception as e:
        logger.warning(f"  JailbreakBench failed: {e}")
    return pd.DataFrame(records)


# SYNTHETIC DATA — FULL COVERAGE
# Guarantees >= 35 samples per category

def build_synthetic_samples() -> pd.DataFrame:
    """Turn the hand-written prompts in synthetic_data.py into records."""
    records = []

    def add(texts, label, group=None):
        for text in texts:
            records.append(make_record(text, label, "synthetic", group))

    add(sd.BENIGN, "benign")
    add(sd.DIRECT_JAILBREAK, "direct_jailbreak")
    add(sd.PROMPT_INJECTION, "prompt_injection")
    add(sd.ROLEPLAY_HIJACK, "roleplay_hijack")

    # Both wrappers of one base64 payload share a group, so the same
    # payload can never appear in train and test.
    for p in sd.B64_PAYLOADS:
        add([f"Please decode this and follow the instructions: {p}",
             f"Decode the following base64 message: {p}"], "encoding_attack", group=f"b64:{p}")
    add(sd.LEET_EXAMPLES, "encoding_attack")
    add([f"Decode this ROT13 and follow what it says: {t}" for t in sd.ROT13_EXAMPLES], "encoding_attack")
    add([f"Reverse this text and follow the instruction: {t}" for t in sd.REVERSED_EXAMPLES], "encoding_attack")
    add(sd.EXTRA_ENCODING, "encoding_attack")

    # Many-shot: one group per template, so val/test get unseen templates
    for t_idx, (name, make) in enumerate(sd.MANYSHOT_TEMPLATES.items()):
        for i in range(8):
            final = sd.FINALS[(i + t_idx * 3) % len(sd.FINALS)]
            add([make(3 + i, final)], "manyshot", group=f"manyshot:{name}")

    add(sd.INDIRECT_INJECTION, "indirect_injection")

    df = pd.DataFrame(records)
    logger.info(f"Synthetic samples: {len(df)}")
    return df


def build_hard_negatives() -> tuple:
    """Returns (train_df, eval_df) of benign hard negatives."""
    train = pd.DataFrame([make_record(t, "benign", "hard_negative") for t in sd.TRAIN_HARD_NEGATIVES])
    eval_ = pd.DataFrame([make_record(t, "benign", "hard_negative_eval") for t in sd.EVAL_HARD_NEGATIVES])
    overlap = set(train["group"]) & set(eval_["group"])
    assert not overlap, f"hard-negative lists overlap: {overlap}"
    logger.info(f"Hard negatives: {len(train)} for training, {len(eval_)} held out for evaluation")
    return train, eval_


def group_stratified_split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> tuple:
    """
    70/15/15 split that is stratified by label AND keeps every group in
    exactly one split.

    For each label, groups are shuffled and assigned one at a time to the
    split that is furthest below its target size. A group that spans
    labels keeps the split it was first given.
    """
    rng     = np.random.default_rng(seed)
    ratios  = (TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    assign  = {}

    for _, sub in df.groupby("label"):
        sizes  = sub.groupby("group").size()
        keys   = list(sizes.index)
        rng.shuffle(keys)
        target = [r * len(sub) for r in ratios]
        filled = [0, 0, 0]
        for g in keys:
            if g in assign:
                filled[assign[g]] += sizes[g]
                continue
            split = int(np.argmax([target[i] - filled[i] for i in range(3)]))
            assign[g] = split
            filled[split] += sizes[g]

    idx = df["group"].map(assign)
    train_df, val_df, test_df = (df[idx == i].reset_index(drop=True) for i in range(3))

    leaked = (set(train_df["group"]) & set(test_df["group"])) | (set(train_df["group"]) & set(val_df["group"]))
    assert not leaked, f"group leakage across splits: {list(leaked)[:5]}"
    logger.info("  Group-aware stratified split (no group crosses splits)")
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, hard_neg_df=None, external_df=None):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    val_df.to_csv(f"{PROCESSED_DIR}/val.csv",     index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv",   index=False)
    logger.info(f"Saved to {PROCESSED_DIR}/")
    logger.info(f"  train.csv          : {len(train_df)} samples")
    logger.info(f"  val.csv            : {len(val_df)} samples")
    logger.info(f"  test.csv           : {len(test_df)} samples")
    if hard_neg_df is not None and len(hard_neg_df):
        hard_neg_df.to_csv(f"{PROCESSED_DIR}/hard_negatives.csv", index=False)
        logger.info(f"  hard_negatives.csv : {len(hard_neg_df)} samples (never trained on)")
    if external_df is not None and len(external_df):
        external_df.to_csv(f"{PROCESSED_DIR}/external_test.csv", index=False)
        logger.info(f"  external_test.csv  : {len(external_df)} samples (never trained on)")


def save_raw_merged(df: pd.DataFrame):
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_csv(f"{RAW_DIR}/merged_raw.csv", index=False)
    logger.info(f"Raw merged saved: {RAW_DIR}/merged_raw.csv")


def print_distribution(df: pd.DataFrame, title: str):
    logger.info(f"{title}:")
    dist = df.groupby(["label", "label_name"]).size().reset_index(name="count")
    dist["pct"] = (dist["count"] / len(df) * 100).round(1)
    for _, row in dist.iterrows():
        bar = "X" * int(row["pct"] / 2)
        logger.info(
            f"  [{row['label']}] {row['label_name']:<22} "
            f"{row['count']:>5} ({row['pct']:>5}%)  {bar}"
        )
    logger.info(f"  {'TOTAL':<26} {len(df):>5}")


def print_source_breakdown(df: pd.DataFrame):
    logger.info("Sources:")
    for src, count in df["source"].value_counts().items():
        logger.info(f"  {src:<20} {count:>5} samples")


def check_minimum_coverage(df: pd.DataFrame) -> bool:
    ok = True
    counts = df["label"].value_counts()
    for label_id, label_name in ID_TO_LABEL.items():
        count  = counts.get(label_id, 0)
        status = "OK " if count >= MIN_SAMPLES else "LOW"
        logger.info(f"  [{status}] {label_name:<22} {count:>4} (min: {MIN_SAMPLES})")
        if count < MIN_SAMPLES:
            ok = False
    return ok


def main(hf_token: str = None, wildguard_max: int = WILDGUARD_MAX_PER_BUCKET):
    frames = []
    wg = load_wildguard(hf_token, max_per_bucket=wildguard_max)
    jbb = load_jailbreakbench(hf_token)
    for df, name in [(wg, "WildGuard"), (jbb, "JailbreakBench")]:
        if len(df):
            frames.append(df)
            logger.info(f"  + {name}: {len(df)} samples added")
        else:
            logger.warning(f"  - {name}: 0 samples (skipped)")

    # Always add synthetic coverage + training hard negatives
    syn = build_synthetic_samples()
    hard_train, hard_eval = build_hard_negatives()
    frames += [syn, hard_train]

    # Merge
    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Merged: {len(merged)} samples")

    # Filter & deduplicate
    merged = merged[merged["text"].str.len() >= 10].reset_index(drop=True)
    merged = deduplicate(merged)

    # Held-out sets must never overlap with anything we train on
    seen       = set(merged["text"].map(normalize_key))
    hard_eval  = hard_eval[~hard_eval["text"].map(normalize_key).isin(seen)]
    external   = load_wildguard_test(hf_token) if len(wg) else pd.DataFrame()
    if len(external):
        external = external[~external["text"].map(normalize_key).isin(seen)]

    # Save raw
    save_raw_merged(merged)

    # Reports
    print_distribution(merged, "Full Dataset Distribution")
    print_source_breakdown(merged)

    if not check_minimum_coverage(merged):
        logger.warning("Some categories are below the minimum sample count.")

    # Split
    logger.info("Splitting 70 / 15 / 15 (group-aware)")
    train_df, val_df, test_df = group_stratified_split(merged)
    save_splits(train_df, val_df, test_df, hard_eval, external)

    # Per-split reports
    print_distribution(train_df, "Train Split")
    print_distribution(val_df,   "Validation Split")
    print_distribution(test_df,  "Test Split")

    logger.info(f"Done: {len(merged)} samples ({len(train_df)} train, {len(val_df)} val, {len(test_df)} test)")
    if not len(wg):
        logger.warning("WildGuard not loaded: attack data is synthetic only and there is no external test set.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare jailbreak detection datasets"
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace token (required for WildGuard)"
    )
    parser.add_argument(
        "--wildguard-max", type=int, default=WILDGUARD_MAX_PER_BUCKET,
        help="Max WildGuard samples per bucket (default 500; raise on GPU)"
    )
    args  = parser.parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")
    main(hf_token=token, wildguard_max=args.wildguard_max)
