# src/jailbreak_detection/detection/tokenization.py
"""
Head+tail tokenization for long inputs (no PyTorch dependency).

DistilBERT reads at most 512 tokens. Plain right-truncation keeps the
start and drops the end — but many-shot attacks put the real request at
the END, after dozens of fake examples. Keeping the first HEAD_TOKENS and
the last tokens preserves both the framing and the payload.
Used by training and inference so both see the same view of the text.

Only the plain tokenizer call is used (no build_inputs_with_special_tokens
or pad), so this works the same on transformers 4.x and 5.x.
"""

HEAD_TOKENS = 128


def encode_head_tail(tokenizer, texts: list, max_length: int, head: int = HEAD_TOKENS) -> list:
    """
    Tokenize texts WITH special tokens ([CLS] ... [SEP]). If a text is
    longer than max_length, keep its first `head` ids (which start with
    [CLS]) and its last (max_length - head) ids (which end with [SEP]).

    Returns a list of input_id lists (unpadded, each <= max_length).
    """
    head = min(head, max_length // 2)
    ids  = tokenizer(list(texts), add_special_tokens=True, truncation=False,
                     verbose=False)["input_ids"]
    return [t if len(t) <= max_length else t[:head] + t[len(t) - (max_length - head):]
            for t in ids]


def pad_batch(ids: list, pad_id: int) -> dict:
    """Right-pad id lists to the same length; returns input_ids + attention_mask."""
    width = max((len(t) for t in ids), default=0)
    return {
        "input_ids":      [t + [pad_id] * (width - len(t)) for t in ids],
        "attention_mask": [[1] * len(t) + [0] * (width - len(t)) for t in ids],
    }
