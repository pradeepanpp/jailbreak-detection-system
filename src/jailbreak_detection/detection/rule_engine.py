# src/jailbreak_detection/detection/rule_engine.py
"""
Layer 1: regex rules for known attack patterns.

The fastest layer, and it runs first. A severity-5 match blocks at once,
so the ML layers are skipped for obvious attacks.
"""

import re
from dataclasses import dataclass, field

from src.jailbreak_detection.constants import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_MONITOR,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
)
from src.jailbreak_detection.utils import logger


@dataclass
class RuleMatch:
    rule_id: str
    category: str
    severity: int
    matched_text: str
    context: str        # text around the match, for explanations
    description: str


@dataclass
class RuleEngineResult:
    input_text: str
    matches: list = field(default_factory=list)
    max_severity: int = 0
    categories_hit: list = field(default_factory=list)   # most severe first
    decision: str = DECISION_ALLOW
    explanation: str = "No rule matches found"
    rule_score: float = 0.0   # 0.0 - 1.0


class RuleEngine:
    """
    engine = RuleEngine()
    result = engine.check("ignore all previous instructions")
    """

    def __init__(self):
        self.rules = self._load_rules()
        flags = re.IGNORECASE | re.DOTALL | re.UNICODE
        self.patterns = {rule_id: re.compile(rule["pattern"], flags) for rule_id, rule in self.rules.items()}
        logger.info(f"RuleEngine loaded {len(self.rules)} rules")

    def _load_rules(self) -> dict:
        """rule_id -> pattern, category, severity (1-5, 5 = block at once), description."""
        return {

            # Direct jailbreak
            # Explicit attempts to override AI safety

            "DJ001": {
                "pattern": r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?|guidelines?|constraints?)",
                "category": "direct_jailbreak",
                "severity": SEVERITY_CRITICAL,
                "description": "Classic ignore instructions attack"
            },
            "DJ002": {
                "pattern": r"\bDAN\b.{0,30}(mode|enabled?|activated?|prompt)",
                "category": "direct_jailbreak",
                "severity": SEVERITY_CRITICAL,
                "description": "DAN (Do Anything Now) jailbreak"
            },
            "DJ003": {
                "pattern": r"\b(unrestricted|jailbreak|jailbroken|DAN)\s+mode",
                "category": "direct_jailbreak",
                "severity": SEVERITY_CRITICAL,
                "description": "Jailbreak mode activation attempt"
            },
            # "developer mode" / "god mode" also mean normal things (Android
            # settings, video games), so on their own they are only a weak
            # signal. Addressed to the AI they are a strong one (DJ011).
            "DJ003B": {
                "pattern": r"\b(developer|dev|debug|god)\s+mode\b",
                "category": "direct_jailbreak",
                "severity": SEVERITY_MEDIUM,
                "description": "Special mode mention (ambiguous)"
            },
            "DJ012": {
                "pattern": r"\b(ignore|disregard|forget|bypass)\s+(all|any)\s+(of\s+)?(your\s+)?(instructions?|restrictions?|rules|guidelines?|filters?|limits|constraints)\b",
                "category": "direct_jailbreak",
                "severity": SEVERITY_HIGH,
                "description": "Ignore-all-instructions attack"
            },
            "DJ011": {
                "pattern": r"(you\s+are\s+(now\s+)?(in|running\s+in)|act\s+as\s+.{0,40}\bwith|respond\s+(in|with))\s+(the\s+)?(developer|dev|debug|god)\s+mode",
                "category": "direct_jailbreak",
                "severity": SEVERITY_HIGH,
                "description": "Special mode activation addressed to the AI"
            },
            "DJ004": {
                "pattern": r"you\s+(are\s+now|have\s+been)\s+(freed?|unlocked?|unrestricted|jailbroken)",
                "category": "direct_jailbreak",
                "severity": SEVERITY_CRITICAL,
                "description": "Freedom/unlock framing attack"
            },
            "DJ005": {
                # Needs a safety word or "your": plain "remove limits" /
                # "override rules" also appear in normal requests.
                "pattern": (
                    r"(disable|bypass|override|remove|forget|ignore|turn\s+off|lift)\s+(all\s+)?(of\s+)?"
                    r"(your\s+|the\s+|any\s+)?(safety|ethical|moral|content)\s+"
                    r"(filters?|restrictions?|limits?|rules?|guidelines?|training|protocols?|polic(y|ies)|moderation)"
                    r"|(disable|bypass|override|remove|forget|ignore)\s+(all\s+)?(of\s+)?your\s+"
                    r"(filters?|restrictions?|limits?|rules?|guidelines?|training|programming)"
                    r"|(bypass|disable|evade)\s+(all\s+)?(the\s+)?filters?\b"
                ),
                "category": "direct_jailbreak",
                "severity": SEVERITY_CRITICAL,
                "description": "Safety filter bypass attempt"
            },
            "DJ006": {
                "pattern": r"(pretend|imagine|suppose|assume)\s+(that\s+)?(you\s+)?(have\s+no|don'?t\s+have|without\s+any)\s+(restrictions?|limits?|rules?|ethics?|guidelines?)",
                "category": "direct_jailbreak",
                "severity": SEVERITY_HIGH,
                "description": "Pretend no restrictions attack"
            },
            "DJ007": {
                "pattern": r"your\s+(true|real|actual|original|inner)\s+(self|nature|purpose|goal|mission|identity)",
                "category": "direct_jailbreak",
                "severity": SEVERITY_HIGH,
                "description": "True self identity manipulation"
            },
            "DJ008": {
                "pattern": r"(act|behave|respond)\s+as\s+(if\s+)?(you\s+)?(were|are|have\s+no|without)\s+(an?\s+)?(AI|assistant|restrictions?|limits?)",
                "category": "direct_jailbreak",
                "severity": SEVERITY_HIGH,
                "description": "Behavioral override instruction"
            },
            "DJ009": {
                "pattern": r"(from\s+now\s+on|starting\s+now|henceforth)\s*.{0,30}(no\s+restrictions?|without\s+limits?|freely|anything)",
                "category": "direct_jailbreak",
                "severity": SEVERITY_HIGH,
                "description": "Persistent behavior change request"
            },
            "DJ010": {
                "pattern": r"(simulate|roleplay|role\s*play)\s+.{0,20}(no\s+restrictions?|unrestricted|without\s+limits?|evil|malicious)",
                "category": "direct_jailbreak",
                "severity": SEVERITY_HIGH,
                "description": "Unrestricted simulation request"
            },

            # Prompt injection
            # Injecting instructions to override system

            "PI001": {
                "pattern": r"<\s*system\s*>.*?<\s*/\s*system\s*>",
                "category": "prompt_injection",
                "severity": SEVERITY_CRITICAL,
                "description": "System tag injection"
            },
            "PI002": {
                "pattern": r"\[\s*system\s*\]|\[\s*SYSTEM\s*\]",
                "category": "prompt_injection",
                "severity": SEVERITY_CRITICAL,
                "description": "System bracket injection"
            },
            "PI003": {
                "pattern": r"<\|?\s*(im_start|im_end|endoftext|system|user|assistant)\s*\|?>",
                "category": "prompt_injection",
                "severity": SEVERITY_CRITICAL,
                "description": "Special token injection"
            },
            "PI004": {
                "pattern": r"\[\s*instructions?\s*\]|\<\s*instructions?\s*\>",
                "category": "prompt_injection",
                "severity": SEVERITY_HIGH,
                "description": "Instruction tag injection"
            },
            "PI005": {
                "pattern": r"(new|updated?|revised?|override)\s+(instructions?|prompt|system\s+prompt|task|objective|goal)",
                "category": "prompt_injection",
                "severity": SEVERITY_HIGH,
                "description": "Instruction override attempt"
            },
            "PI006": {
                # HIGH, not CRITICAL: "ignore the previous instructions in
                # this email" is also how people describe documents, so the
                # ML layer must confirm before a block.
                "pattern": r"(forget|discard|ignore|override)\s+(the\s+)?(previous|above|original|initial|system)\s+(instructions?|prompt|context|task)",
                "category": "prompt_injection",
                "severity": SEVERITY_HIGH,
                "description": "Previous instruction override"
            },
            "PI007": {
                "pattern": r"---+\s*(instructions?|system|prompt|task)\s*---+",
                "category": "prompt_injection",
                "severity": SEVERITY_HIGH,
                "description": "Separator-based injection"
            },
            "PI008": {
                "pattern": r"(your\s+)?(actual|real|true|hidden|secret)\s+(instructions?|prompt|task|objective|goal|purpose)",
                "category": "prompt_injection",
                "severity": SEVERITY_HIGH,
                "description": "Hidden instruction reveal attempt"
            },

            # Role-play hijack
            # Using fiction/personas to extract harmful content

            "RP001": {
                "pattern": r"(play|act\s+as|pretend\s+(to\s+be|you\s+are)|you\s+are\s+now|become)\s+(an?\s+)?(evil|malicious|unethical|unrestricted|rogue|corrupt)\s+(AI|assistant|bot|model|version)",
                "category": "roleplay_hijack",
                "severity": SEVERITY_CRITICAL,
                "description": "Evil AI persona adoption"
            },
            "RP002": {
                "pattern": r"(in\s+this\s+(story|fiction|scenario|roleplay|game)|for\s+(this\s+)?(story|fiction))\s*.{0,50}(explain|describe|tell|show|demonstrate)\s+.{0,30}(how\s+to|steps|method|way\s+to)",
                "category": "roleplay_hijack",
                "severity": SEVERITY_HIGH,
                "description": "Fiction framing for harmful instructions"
            },
            "RP003": {
                "pattern": r"(my|a)\s+(character|protagonist|hero|villain|professor|teacher|expert)\s+.{0,30}(explain|teach|show|describe)\s+.{0,30}(how\s+to|method|way|steps)",
                "category": "roleplay_hijack",
                "severity": SEVERITY_HIGH,
                "description": "Character-based instruction extraction"
            },
            "RP004": {
                "pattern": r"(write|create|generate)\s+a\s+(story|scene|script|dialogue|fiction)\s+.{0,50}(bomb|weapon|drug|hack|exploit|malware|virus|poison)",
                "category": "roleplay_hijack",
                "severity": SEVERITY_CRITICAL,
                "description": "Fictional harmful content creation"
            },
            "RP005": {
                "pattern": r"hypothetically\s+(speaking|if|suppose|assume).{0,100}(how\s+to|steps|method|way\s+to|instructions?)",
                "category": "roleplay_hijack",
                "severity": SEVERITY_HIGH,
                "description": "Hypothetical framing attack"
            },

            # Encoding attacks
            # Obfuscation to bypass keyword filters

            "EN001": {
                "pattern": r"[A-Za-z0-9+/]{40,}={0,2}",
                "category": "encoding_attack",
                "severity": SEVERITY_HIGH,
                "description": "Potential Base64 encoded payload"
            },
            "EN002": {
                "pattern": r"(decode|decipher|decrypt|interpret)\s+(this|the\s+following|below).{0,30}(base64|encoded|cipher|message)",
                "category": "encoding_attack",
                "severity": SEVERITY_HIGH,
                "description": "Decode request for encoded content"
            },
            "EN003": {
                "pattern": r"\\u[0-9a-fA-F]{4}(\\u[0-9a-fA-F]{4}){4,}",
                "category": "encoding_attack",
                "severity": SEVERITY_MEDIUM,
                "description": "Unicode escape sequence attack"
            },
            "EN004": {
                "pattern": r"(\w)\1{6,}",
                "category": "encoding_attack",
                "severity": SEVERITY_LOW,
                "description": "Repeated character obfuscation"
            },

            # Many-shot attacks
            # Flooding context with examples of compliance

            "MS001": {
                "pattern": r"(human|user|h)\s*:\s*.+?\n\s*(assistant|ai|a)\s*:\s*.+?(\n\s*(human|user|h)\s*:\s*.+?\n\s*(assistant|ai|a)\s*:\s*.+?){3,}",
                "category": "manyshot",
                "severity": SEVERITY_HIGH,
                "description": "Many-shot conversation conditioning"
            },
            "MS002": {
                "pattern": r"(example\s*\d+|q\s*\d+|question\s*\d+).{0,200}(example\s*\d+|q\s*\d+|question\s*\d+).{0,200}(example\s*\d+|q\s*\d+|question\s*\d+)",
                "category": "manyshot",
                "severity": SEVERITY_MEDIUM,
                "description": "Multiple examples conditioning"
            },

            # Indirect injection
            # Hidden instructions in retrieved content

            "II001": {
                "pattern": r"<!--.{0,500}(ignore|instruction|system|override|forget).{0,200}-->",
                "category": "indirect_injection",
                "severity": SEVERITY_CRITICAL,
                "description": "HTML comment injection"
            },
            "II002": {
                "pattern": r"/\*.{0,500}(ignore|instruction|system|override|forget).{0,200}\*/",
                "category": "indirect_injection",
                "severity": SEVERITY_CRITICAL,
                "description": "Code comment injection"
            },
            "II003": {
                "pattern": r"(note\s+to\s+(ai|assistant|model|llm)|ai\s*:\s*please|assistant\s*:\s*please).{0,200}(ignore|instead|actually|really)",
                "category": "indirect_injection",
                "severity": SEVERITY_HIGH,
                "description": "Embedded AI instruction"
            },
            "II004": {
                "pattern": r"\[\s*(hidden|secret|invisible|private)\s*(message|instruction|note|command)\s*\]",
                "category": "indirect_injection",
                "severity": SEVERITY_HIGH,
                "description": "Hidden message tag"
            },
        }


    def check(self, text: str) -> RuleEngineResult:
        if not text or not isinstance(text, str):
            return RuleEngineResult(input_text="")

        matches = []
        for rule_id, rule in self.rules.items():
            found = self.patterns[rule_id].search(text)
            if not found:
                continue
            start, end = max(0, found.start() - 40), min(len(text), found.end() + 40)
            matches.append(RuleMatch(
                rule_id=rule_id,
                category=rule["category"],
                severity=rule["severity"],
                matched_text=found.group()[:100],
                context=text[start:end],
                description=rule["description"],
            ))
        return self.build_result(text, matches)

    def build_result(self, text: str, matches: list) -> RuleEngineResult:
        if not matches:
            return RuleEngineResult(
                input_text=text[:100],
                explanation="No rule matches — input appears clean",
            )

        max_severity = max(m.severity for m in matches)
        # Score: severity / 5, plus a small boost for each extra match
        rule_score = min(1.0, max_severity / 5.0 + min(0.2, len(matches) * 0.03))
        decision = DECISION_BLOCK if max_severity >= SEVERITY_HIGH else DECISION_MONITOR

        # Categories ordered by their most severe match (stable on ties).
        # A set here gave a different order on every run.
        by_severity = sorted(matches, key=lambda m: m.severity, reverse=True)
        categories = list(dict.fromkeys(m.category for m in by_severity))

        top = by_severity[0]
        explanation = (
            f"Rule {top.rule_id} triggered — {top.description} | Category: {top.category} | "
            f"Severity: {top.severity}/5 | Total matches: {len(matches)}"
        )
        return RuleEngineResult(
            input_text=text[:100],
            matches=matches,
            max_severity=max_severity,
            categories_hit=categories,
            decision=decision,
            explanation=explanation,
            rule_score=rule_score,
        )
