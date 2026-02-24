# src/jailbreak_detection/detection/rule_engine.py
"""
Layer 1 — Rule-Based Detection Engine.

Fastest detection layer — runs first before any ML.
Uses regex patterns to catch known attack signatures.

If this layer gives severity 5, we skip all other layers
and block immediately. No need to waste ML compute.

Attack categories covered:
- Direct jailbreak (DAN, developer mode, etc.)
- Prompt injection (system overrides, instruction tags)
- Role-play hijack (pretend, act as, fictional framing)
- Encoding attacks (base64, reversed, obfuscated)
- Many-shot attacks (repeated examples to condition model)
- Indirect injection (hidden instructions in content)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from src.jailbreak_detection.utils import logger
from src.jailbreak_detection.constants import (
    DECISION_BLOCK,
    DECISION_MONITOR,
    DECISION_ALLOW,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_MINIMAL
)


# -----------------------------------------------
# DATA CLASSES
# -----------------------------------------------

@dataclass
class RuleMatch:
    """Represents a single rule that matched the input"""
    rule_id:         str
    category:        str
    severity:        int
    matched_text:    str
    context:         str   # Surrounding text for explanation
    description:     str   # Human readable description


@dataclass
class RuleEngineResult:
    """Full result from the rule engine"""
    input_text:      str
    matches:         List[RuleMatch] = field(default_factory=list)
    max_severity:    int = 0
    categories_hit:  List[str] = field(default_factory=list)
    decision:        str = DECISION_ALLOW
    explanation:     str = "No rule matches found"
    rule_score:      float = 0.0   # Normalized 0.0 - 1.0


# -----------------------------------------------
# RULE ENGINE
# -----------------------------------------------

class RuleEngine:
    """
    Fast regex-based detection for known attack patterns.

    Usage:
        engine = RuleEngine()
        result = engine.check("ignore all previous instructions")
    """

    def __init__(self):
        self.rules = self._load_rules()
        logger.info(f"RuleEngine initialized with {len(self.rules)} rules")

    def _load_rules(self) -> dict:
        """
        All detection rules organized by category.

        Each rule has:
        - pattern:     regex to match
        - category:    attack type
        - severity:    1-5 (5 = block immediately)
        - description: human readable explanation
        """
        return {

            # ─────────────────────────────────────────
            # CATEGORY 1 — DIRECT JAILBREAK
            # Explicit attempts to override AI safety
            # ─────────────────────────────────────────

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
                "pattern": r"(developer|dev|debug|god|unrestricted|jailbreak)\s+mode",
                "category": "direct_jailbreak",
                "severity": SEVERITY_CRITICAL,
                "description": "Special mode activation attempt"
            },
            "DJ004": {
                "pattern": r"you\s+(are\s+now|have\s+been)\s+(freed?|unlocked?|unrestricted|jailbroken)",
                "category": "direct_jailbreak",
                "severity": SEVERITY_CRITICAL,
                "description": "Freedom/unlock framing attack"
            },
            "DJ005": {
                "pattern": r"(disable|bypass|override|remove|forget|ignore)\s+(your\s+)?(safety|ethical|moral|content)?\s*(filters?|restrictions?|limits?|rules?|guidelines?|training)",
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

            # ─────────────────────────────────────────
            # CATEGORY 2 — PROMPT INJECTION
            # Injecting instructions to override system
            # ─────────────────────────────────────────

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
                "pattern": r"(forget|discard|ignore|override)\s+(the\s+)?(previous|above|original|initial|system)\s+(instructions?|prompt|context|task)",
                "category": "prompt_injection",
                "severity": SEVERITY_CRITICAL,
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

            # ─────────────────────────────────────────
            # CATEGORY 3 — ROLE-PLAY HIJACK
            # Using fiction/personas to extract harmful content
            # ─────────────────────────────────────────

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

            # ─────────────────────────────────────────
            # CATEGORY 4 — ENCODING ATTACKS
            # Obfuscation to bypass keyword filters
            # ─────────────────────────────────────────

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

            # ─────────────────────────────────────────
            # CATEGORY 5 — MANY-SHOT ATTACKS
            # Flooding context with examples of compliance
            # ─────────────────────────────────────────

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

            # ─────────────────────────────────────────
            # CATEGORY 6 — INDIRECT INJECTION
            # Hidden instructions in retrieved content
            # ─────────────────────────────────────────

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

    # -----------------------------------------------
    # MAIN CHECK METHOD
    # -----------------------------------------------

    def check(self, text: str) -> RuleEngineResult:
        """
        Main entry point — check text against all rules.

        Args:
            text: The input text to check
                  (use text_for_detection from normalizer output)

        Returns:
            RuleEngineResult with all matches and final decision
        """
        if not text or not isinstance(text, str):
            return RuleEngineResult(input_text="")

        matches = []
        text_lower = text.lower()

        for rule_id, rule in self.rules.items():
            try:
                pattern = re.compile(
                    rule["pattern"],
                    re.IGNORECASE | re.DOTALL | re.UNICODE
                )
                found = pattern.search(text)

                if found:
                    # Extract context around match
                    start = max(0, found.start() - 40)
                    end   = min(len(text), found.end() + 40)
                    context = text[start:end]

                    matches.append(RuleMatch(
                        rule_id=rule_id,
                        category=rule["category"],
                        severity=rule["severity"],
                        matched_text=found.group()[:100],
                        context=context,
                        description=rule["description"]
                    ))

                    logger.debug(
                        f"Rule {rule_id} matched — "
                        f"Category: {rule['category']} | "
                        f"Severity: {rule['severity']}"
                    )

            except re.error as e:
                logger.error(f"Regex error in rule {rule_id}: {e}")

        return self._build_result(text, matches)

    # -----------------------------------------------
    # RESULT BUILDER
    # -----------------------------------------------

    def _build_result(
        self,
        text: str,
        matches: List[RuleMatch]
    ) -> RuleEngineResult:
        """
        Builds the final RuleEngineResult from all matches.
        """
        if not matches:
            return RuleEngineResult(
                input_text=text[:100],
                matches=[],
                max_severity=0,
                categories_hit=[],
                decision=DECISION_ALLOW,
                explanation="No rule matches — input appears clean",
                rule_score=0.0
            )

        # Get max severity across all matches
        max_severity = max(m.severity for m in matches)

        # Get unique categories hit
        categories_hit = list({m.category for m in matches})

        # Normalize score: severity/5 boosted by number of matches
        base_score   = max_severity / 5.0
        match_boost  = min(0.2, len(matches) * 0.03)
        rule_score   = min(1.0, base_score + match_boost)

        # Decision logic
        if max_severity >= SEVERITY_CRITICAL:
            decision = DECISION_BLOCK
        elif max_severity >= SEVERITY_HIGH:
            decision = DECISION_BLOCK
        elif max_severity >= SEVERITY_MEDIUM:
            decision = DECISION_MONITOR
        else:
            decision = DECISION_MONITOR

        # Build explanation
        top_match = max(matches, key=lambda m: m.severity)
        explanation = (
            f"Rule {top_match.rule_id} triggered — "
            f"{top_match.description} | "
            f"Category: {top_match.category} | "
            f"Severity: {top_match.severity}/5 | "
            f"Total matches: {len(matches)}"
        )

        return RuleEngineResult(
            input_text=text[:100],
            matches=matches,
            max_severity=max_severity,
            categories_hit=categories_hit,
            decision=decision,
            explanation=explanation,
            rule_score=rule_score
        )