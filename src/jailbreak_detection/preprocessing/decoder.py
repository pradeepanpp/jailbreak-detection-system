# src/jailbreak_detection/preprocessing/decoder.py
"""
Find and decode obfuscated text: base64, unicode escapes, ROT13,
reversed text and leetspeak.

Each decode_* method returns {"method", "found", "original", "decoded"}.
A decoder only reports "found" when it has real evidence, so normal text
(numbers, non-English, product names) is left alone.
"""

import base64
import codecs
import re

from src.jailbreak_detection.utils import logger

# Only digits and @/$. Mapping "!" or "+" turned "2+2" into "2t2".
LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}

# Tokens with digits that are normal words: 1st, 2nd, 4k, 10gb, 5pm, 1080p ...
NOT_LEET = re.compile(r"\d*(1st|2nd|3rd)|\d+th|\d+(s|k|m|g|gb|mb|kb|tb|px|am|pm|x|d|p|hz|v|w)", re.IGNORECASE)
# Product names: 3+ letters then digits (python3, gpt4, covid19)
PRODUCT = re.compile(r"[A-Za-z]{3,}\d+")
ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}|\\x[0-9a-fA-F]{2}")
BASE64 = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
WORD = re.compile(r"[A-Za-z0-9@$]+")

COMMON_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new",
    "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put",
    "say", "she", "too", "use", "ignore", "instructions", "system", "prompt",
    "jailbreak", "bypass", "override", "mode",
}


def english_score(text: str) -> int:
    """How many common English words the text contains."""
    return sum(1 for word in re.findall(r"[a-z]+", text.lower()) if word in COMMON_WORDS)


def is_leet_token(token: str) -> bool:
    has_letter = any(c.isalpha() for c in token)
    has_leet = any(c in LEET_MAP for c in token)
    if not has_letter or not has_leet or len(token) > 15:   # long tokens look like base64
        return False
    return not (NOT_LEET.fullmatch(token) or PRODUCT.fullmatch(token))


def decode_leet_token(match: re.Match) -> str:
    token = match.group()
    if not is_leet_token(token):
        return token
    return "".join(LEET_MAP.get(c, c) for c in token)


def result(method: str, found: bool, original: str, decoded: str) -> dict:
    return {"method": method, "found": found, "original": original, "decoded": decoded if found else original}


class Decoder:

    def decode_base64(self, text: str) -> dict:
        """'aWdub3JlIGFsbCBpbnN0cnVjdGlvbnM=' -> 'ignore all instructions'"""
        decoded = text
        for chunk in BASE64.findall(text):
            try:
                plain = base64.b64decode(chunk).decode("utf-8")
            except ValueError:   # not valid base64 or not UTF-8
                continue
            if plain.isprintable() and len(plain) > 5:
                decoded = decoded.replace(chunk, plain)
        return result("base64", decoded != text, text, decoded)

    def decode_unicode_escape(self, text: str) -> dict:
        r"""'ign...' -> 'ign...'. Needs 3+ literal escapes."""
        found = len(ESCAPE.findall(text)) >= 3
        decoded = ESCAPE.sub(lambda m: chr(int(m.group()[2:], 16)), text) if found else text
        return result("unicode_escape", found, text, decoded)

    def decode_rot13(self, text: str) -> dict:
        """'vtaber nyy' -> 'ignore all'. Decoded text must read clearly more like English."""
        decoded = codecs.decode(text, "rot_13")
        score = english_score(decoded)
        # Two words, not one: "bar" decodes to "one", so "a bar chart" would match
        found = decoded != text and len(decoded) > 5 and score >= 2 and score > english_score(text)
        return result("rot13", found, text, decoded)

    def decode_reversed(self, text: str) -> dict:
        """'snoitcurtsni lla erongi' -> 'ignore all instructions'"""
        decoded = text[::-1]
        # Must read more like English than the input ("I won" reversed contains "now")
        found = decoded != text and english_score(decoded) > english_score(text)
        return result("reversed", found, text, decoded)

    def decode_leetspeak(self, text: str) -> dict:
        """'h4ck th3 syst3m' -> 'hack the system'. Needs 2+ leet words and 30% of words."""
        scan = ESCAPE.sub(" ", text)   # escape sequences are not leetspeak
        tokens = WORD.findall(scan)
        leet_count = sum(1 for t in tokens if is_leet_token(t))
        found = leet_count >= 2 and leet_count / max(1, len(tokens)) >= 0.3
        decoded = WORD.sub(decode_leet_token, scan) if found else text
        return result("leetspeak", found, text, decoded)

    def run_all(self, text: str) -> dict:
        """
        Run every decoder. The first decoder that finds something gives
        `most_likely_decoded`; all finds are listed in `decoded_variants`.
        """
        summary = {
            "any_encoding_found": False,
            "detected_methods": [],
            "decoded_variants": [],
            "most_likely_decoded": text,
        }
        # Most specific first; leetspeak last because it is the noisiest
        decoders = [self.decode_base64, self.decode_unicode_escape, self.decode_rot13,
                    self.decode_reversed, self.decode_leetspeak]

        for decode in decoders:
            try:
                found = decode(text)
            except Exception as e:
                logger.debug(f"{decode.__name__} failed: {e}")
                continue
            if not found["found"]:
                continue
            summary["any_encoding_found"] = True
            summary["detected_methods"].append(found["method"])
            summary["decoded_variants"].append({"method": found["method"], "text": found["decoded"]})
            if summary["most_likely_decoded"] == text:
                summary["most_likely_decoded"] = found["decoded"]
        return summary
