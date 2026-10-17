# src/jailbreak_detection/detection/tokenization.py
"""
Head + tail tokenization for long inputs.

DistilBERT reads at most 512 tokens. Normal truncation keeps the start
and drops the end, but many-shot attacks put the real request at the end.
So for long texts we keep the first HEAD_TOKENS tokens and the last ones.
Used in both training and inference. Needs no PyTorch.
"""

HEAD_TOKENS = 128


def encode_head_tail(tokenizer, texts: list, max_length: int, head: int = HEAD_TOKENS) -> list:
    """Token ids for each text ([CLS] ... [SEP]), at most max_length long."""
    head = min(head, max_length // 2)
    tail = max_length - head
    all_ids = tokenizer(list(texts), add_special_tokens=True, truncation=False, verbose=False)["input_ids"]

    result = []
    for ids in all_ids:
        if len(ids) > max_length:
            ids = ids[:head] + ids[-tail:]   # starts with [CLS], ends with [SEP]
        result.append(ids)
    return result


def pad_batch(ids: list, pad_id: int) -> dict:
    """Right-pad id lists to the same length and build the attention mask."""
    width = max((len(x) for x in ids), default=0)
    input_ids, attention_mask = [], []
    for x in ids:
        padding = width - len(x)
        input_ids.append(x + [pad_id] * padding)
        attention_mask.append([1] * len(x) + [0] * padding)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
