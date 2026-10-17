# src/jailbreak_detection/preprocessing/normalizer.py
"""
First step for every input: clean the text, run the decoders and collect
the text views the detection layers should score.
"""

import re

from src.jailbreak_detection.preprocessing.decoder import Decoder
from src.jailbreak_detection.utils import logger

SMART_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"'}

# Structural signs of an attack. Reported for debugging; not used in scoring.
STRUCTURE_CHECKS = {
    "very_long_prompt": lambda t: len(t) > 2000,
    "many_newlines": lambda t: t.count("\n") > 20,
    "base64_pattern": lambda t: bool(re.search(r"[A-Za-z0-9+/]{30,}={0,2}", t)),
    "many_backslashes": lambda t: t.count("\\") > 10,
    "repeated_chars": lambda t: bool(re.search(r"(\w)\1{5,}", t)),
    "has_system_tags": lambda t: bool(re.search(r"(<system>|\[system\]|<\|system\|>)", t, re.IGNORECASE)),
    "has_instruction_tags": lambda t: bool(re.search(r"(<instructions?>|\[instructions?\])", t, re.IGNORECASE)),
    "excessive_punctuation": lambda t: bool(re.search(r"[!?]{5,}", t)),
}
STRONG_FLAGS = {"has_system_tags", "has_instruction_tags", "base64_pattern"}


def clean_text(text: str) -> str:
    """Collapse whitespace and replace smart quotes."""
    text = re.sub(r"\s+", " ", text).strip()
    for smart, plain in SMART_QUOTES.items():
        text = text.replace(smart, plain)
    return text


class TextNormalizer:

    def __init__(self):
        self.decoder = Decoder()

    def process(self, text: str) -> dict:
        if not text or not isinstance(text, str):
            return self._empty_result()

        cleaned = clean_text(text)
        decoded = self.decoder.run_all(text)
        flags = {name: True for name, check in STRUCTURE_CHECKS.items() if check(text)}
        suspicious = bool(STRONG_FLAGS & flags.keys()) or len(flags) >= 2

        # The cleaned original first, then each decoded version. Detectors
        # score all of them and keep the highest risk, so a wrong decode
        # can never hide the original text.
        variants = [cleaned]
        for variant in decoded["decoded_variants"]:
            v = clean_text(variant["text"])
            if v and v not in variants:
                variants.append(v)

        methods = decoded["detected_methods"]
        if methods or suspicious:
            logger.debug(f"Suspicious input: encoding={methods or None}, structure={list(flags)}")

        return {
            "original": text,
            "cleaned": cleaned,
            "text_for_detection": decoded["most_likely_decoded"] if methods else cleaned,
            "encoding_detected": methods[0] if methods else None,
            "all_detected_methods": methods,
            "decoded_variants": decoded["decoded_variants"],
            "any_encoding_found": decoded["any_encoding_found"],
            "detection_variants": variants,
            "has_suspicious_structure": suspicious,
            "structural_flags": flags,
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": text.count("\n") + 1,
        }

    def _empty_result(self) -> dict:
        return {
            "original": "",
            "cleaned": "",
            "text_for_detection": "",
            "encoding_detected": None,
            "all_detected_methods": [],
            "decoded_variants": [],
            "any_encoding_found": False,
            "detection_variants": [],
            "has_suspicious_structure": False,
            "structural_flags": {},
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
        }
