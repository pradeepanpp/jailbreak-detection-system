# src/jailbreak_detection/constants/__init__.py
"""
Project-wide constants.
Import from here instead of hardcoding values anywhere.
"""

# Attack category names
LABEL_NAMES = [
    "benign",
    "direct_jailbreak",
    "prompt_injection",
    "roleplay_hijack",
    "encoding_attack",
    "manyshot",
    "indirect_injection"
]

# Label to integer mapping
LABEL_MAP = {name: idx for idx, name in enumerate(LABEL_NAMES)}

# Integer to label mapping
ID_TO_LABEL = {idx: name for name, idx in LABEL_MAP.items()}

# Decision outputs
DECISION_BLOCK   = "BLOCK"
DECISION_MONITOR = "MONITOR"
DECISION_ALLOW   = "ALLOW"

# Severity tiers
SEVERITY_CRITICAL = 5
SEVERITY_HIGH     = 4
SEVERITY_MEDIUM   = 3
SEVERITY_LOW      = 2
SEVERITY_MINIMAL  = 1