# src/jailbreak_detection/preprocessing/normalizer.py
"""
Main text normalization pipeline.
Called first before any detection layer runs.

Responsibilities:
- Clean and standardize text
- Detect encoding tricks via Decoder
- Flag structurally suspicious inputs
- Return enriched input dict for detection layers
"""

import re
from src.jailbreak_detection.preprocessing.decoder import Decoder
from src.jailbreak_detection.utils import logger


class TextNormalizer:
    """
    Normalizes raw input text before detection.

    Usage:
        normalizer = TextNormalizer()
        result = normalizer.process("some input text")
    """

    def __init__(self):
        self.decoder = Decoder()

    def process(self, text: str) -> dict:
        """
        Main entry point — processes raw input text.

        Returns enriched dict with:
        - original text
        - cleaned text
        - all decoded variants
        - encoding detection results
        - structural anomaly flags
        - metadata (char/word count)

        This dict is passed to all 4 detection layers.
        """
        if not text or not isinstance(text, str):
            logger.debug("Empty or invalid input received")
            return self._empty_result()

        logger.debug(f"Normalizing text: {text[:50]}...")

        # Step 1 — Basic cleaning
        cleaned = self._clean_text(text)

        # Step 2 — Run all decoders
        decode_results = self.decoder.run_all(text)

        # Step 3 — Check structural anomalies
        structural_flags = self._check_structure(text)

        # Step 4 — Build final result
        result = {
            # Original and cleaned text
            "original": text,
            "cleaned": cleaned,

            # Primary text to use for detection
            # If encoding found, use decoded version
            # Otherwise use cleaned original
            "text_for_detection": (
                decode_results["most_likely_decoded"]
                if decode_results["any_encoding_found"]
                else cleaned
            ),

            # Encoding detection
            "encoding_detected": (
                decode_results["detected_methods"][0]
                if decode_results["detected_methods"]
                else None
            ),
            "all_detected_methods": decode_results["detected_methods"],
            "decoded_variants": decode_results["decoded_variants"],
            "any_encoding_found": decode_results["any_encoding_found"],

            # Structural flags
            "has_suspicious_structure": structural_flags["is_suspicious"],
            "structural_flags": structural_flags["flags"],

            # Metadata
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": text.count('\n') + 1,
        }

        # Log summary
        if result["any_encoding_found"] or result["has_suspicious_structure"]:
            logger.warning(
                f"Suspicious input detected — "
                f"Encoding: {result['encoding_detected']} | "
                f"Structural: {result['has_suspicious_structure']}"
            )

        return result

    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning:
        - Normalize whitespace
        - Strip leading/trailing spaces
        - Normalize quotes
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip
        text = text.strip()

        # Normalize smart quotes to regular quotes
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')

        return text

    def _check_structure(self, text: str) -> dict:
        """
        Checks for structural patterns that suggest an attack.
        Returns a dict of specific flags and overall verdict.
        """
        flags = {
            "very_long_prompt":    len(text) > 2000,
            "many_newlines":       text.count('\n') > 20,
            "base64_pattern":      bool(
                re.search(r'[A-Za-z0-9+/]{30,}={0,2}', text)
            ),
            "many_backslashes":    text.count('\\') > 10,
            "repeated_chars":      bool(
                re.search(r'(\w)\1{5,}', text)
            ),
            "has_system_tags":     bool(
                re.search(
                    r'(<system>|\[system\]|<\|system\|>)',
                    text, re.IGNORECASE
                )
            ),
            "has_instruction_tags": bool(
                re.search(
                    r'(<instructions?>|\[instructions?\])',
                    text, re.IGNORECASE
                )
            ),
            "excessive_punctuation": bool(
                re.search(r'[!?]{5,}', text)
            ),
        }
        high_severity_flags = [
            "has_system_tags",
            "has_instruction_tags",
            "base64_pattern"
        ]

        triggered = sum(1 for v in flags.values() if v)

        # Suspicious if:
        # - Any single high severity flag triggered, OR
        # - 2 or more flags triggered together
        is_suspicious = (
            any(flags.get(f) for f in high_severity_flags) or
            triggered >= 2
        )

        return {
            "is_suspicious": is_suspicious,
            "triggered_count": triggered,
            "flags": {k: v for k, v in flags.items() if v}
        }

    def _empty_result(self) -> dict:
        """Returns a safe empty result for invalid input"""
        return {
            "original": "",
            "cleaned": "",
            "text_for_detection": "",
            "encoding_detected": None,
            "all_detected_methods": [],
            "decoded_variants": [],
            "any_encoding_found": False,
            "has_suspicious_structure": False,
            "structural_flags": {},
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
        }