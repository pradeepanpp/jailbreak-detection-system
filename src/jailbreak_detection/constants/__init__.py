# src/jailbreak_detection/constants/__init__.py

LABEL_NAMES = [
    "benign",
    "direct_jailbreak",
    "prompt_injection",
    "roleplay_hijack",
    "encoding_attack",
    "manyshot",
    "indirect_injection",
]
LABEL_MAP = {name: i for i, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}

DECISION_BLOCK = "BLOCK"
DECISION_MONITOR = "MONITOR"
DECISION_ALLOW = "ALLOW"

SEVERITY_CRITICAL = 5
SEVERITY_HIGH = 4
SEVERITY_MEDIUM = 3
SEVERITY_LOW = 2
SEVERITY_MINIMAL = 1
