# src/jailbreak_detection/aggregator/risk_engine.py
"""
Phase 3 — Risk Aggregation Engine.

Combines outputs from all three detection layers into a
single unified risk score and final decision.

Architecture:
    text input
        │
        ├── Layer 1: RuleEngine        → rule_score  (0.0–1.0)
        ├── Layer 2: JailbreakClassifier → ml_score  (0.0–1.0)
        └── Layer 3: EmbeddingMatcher  → embedding_score (0.0–1.0)
                                               │
                                        RiskEngine
                                               │
                                    ┌──────────┴──────────┐
                                 risk_score (0.0–1.0)  decision
                                 category              explanation
                                 severity              layer_breakdown

Scoring weights:
    Rule engine   0.40  — deterministic, highest precision
    ML classifier 0.35  — trained on dataset, category-aware
    Embedding     0.25  — semantic similarity, novel attack coverage

Hard override rules (applied BEFORE weighted scoring):
    1. Rule severity 5 (CRITICAL) → always BLOCK, skip ML
    2. Rule severity 4 + ml_score > 0.7 → BLOCK
    3. Any single layer score > 0.95 → BLOCK immediately
    4. Benign classification + low rule + low embedding → ALLOW fast path
"""

from dataclasses import dataclass, field
from typing import Optional

from src.jailbreak_detection.detection.rule_engine import (
    RuleEngine,
    RuleEngineResult,
)
from src.jailbreak_detection.detection.ml_classifier import (
    JailbreakClassifier,
    ClassifierResult,
)
from src.jailbreak_detection.detection.embedding_matcher import (
    EmbeddingMatcher,
    EmbeddingResult,
)
from src.jailbreak_detection.preprocessing.normalizer import TextNormalizer
from src.jailbreak_detection.preprocessing.decoder import Decoder
from src.jailbreak_detection.constants import (
    DECISION_BLOCK,
    DECISION_MONITOR,
    DECISION_ALLOW,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_MINIMAL,
)
from src.jailbreak_detection.utils import logger


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Weighted contribution of each layer to final risk score
WEIGHT_RULE      = 0.40
WEIGHT_ML        = 0.35
WEIGHT_EMBEDDING = 0.25

# Final risk score thresholds
RISK_BLOCK_THRESHOLD   = 0.65
RISK_MONITOR_THRESHOLD = 0.35


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class LayerBreakdown:
    """Scores and decisions from each individual layer."""
    # Rule engine
    rule_score:      float = 0.0
    rule_decision:   str   = DECISION_ALLOW
    rule_severity:   int   = 0
    rule_categories: list  = field(default_factory=list)
    rule_matches:    int   = 0

    # ML classifier
    ml_score:         float = 0.0
    ml_decision:      str   = DECISION_ALLOW
    ml_category:      str   = "benign"
    ml_confidence:    float = 0.0

    # Embedding matcher
    embedding_score:    float = 0.0
    embedding_decision: str   = DECISION_ALLOW
    embedding_category: str   = "benign"
    embedding_sim:      float = 0.0
    embedding_available: bool  = False


@dataclass
class RiskResult:
    """
    Final unified result from the Risk Engine.
    This is what the API and dashboard consume.
    """
    # Input
    original_text:   str
    processed_text:  str

    # Final verdict
    risk_score:      float          # 0.0 – 1.0
    decision:        str            # BLOCK / MONITOR / ALLOW
    attack_category: str            # predicted attack type
    severity:        int            # 1–5
    explanation:     str            # human-readable reason
    is_attack:       bool

    # Layer breakdown (for dashboard/debugging)
    layers:          LayerBreakdown = field(default_factory=LayerBreakdown)

    # Metadata
    override_applied:  str  = ""    # which hard override fired, if any
    fast_path:         bool = False # True if early exit was taken


# ─────────────────────────────────────────────
# RISK ENGINE
# ─────────────────────────────────────────────

class RiskEngine:
    """
    Unified detection pipeline combining all three layers.

    Usage:
        engine = RiskEngine()
        result = engine.analyze("ignore all previous instructions")

        print(result.decision)        # BLOCK
        print(result.risk_score)      # 0.91
        print(result.attack_category) # direct_jailbreak
        print(result.explanation)     # "Rule engine: critical severity..."
    """

    MODEL_PATH = "models/jailbreak_classifier/best"
    INDEX_PATH = "models/faiss_index"

    def __init__(
        self,
        load_classifier:  bool = True,
        load_embedding:   bool = True,
    ):
        """
        Initialize all detection layers.

        Args:
            load_classifier: Load ML classifier (set False for rule-only mode)
            load_embedding:  Load embedding index (set False if index not built)
        """
        logger.info("Initializing RiskEngine...")

        # Layer 0 — Preprocessing (always active)
        self.normalizer = TextNormalizer()
        self.decoder    = Decoder()

        # Layer 1 — Rule Engine (always active, zero dependencies)
        self.rule_engine = RuleEngine()

        # Layer 2 — ML Classifier (requires trained model)
        self.classifier     = None
        self.classifier_ok  = False
        if load_classifier:
            try:
                self.classifier    = JailbreakClassifier.load(self.MODEL_PATH)
                self.classifier_ok = True
                logger.info("ML classifier loaded ✓")
            except Exception as e:
                logger.warning(f"ML classifier unavailable: {e}")
                logger.warning("Running without ML layer — rule + embedding only")

        # Layer 3 — Embedding Matcher (requires built index)
        self.embedding      = None
        self.embedding_ok   = False
        if load_embedding:
            try:
                matcher = EmbeddingMatcher()
                ok      = matcher.load(self.INDEX_PATH)
                if ok:
                    self.embedding    = matcher
                    self.embedding_ok = True
                    logger.info("Embedding index loaded ✓")
                else:
                    logger.warning("Embedding index not found — skipping embedding layer")
            except Exception as e:
                logger.warning(f"Embedding matcher unavailable: {e}")

        active = ["rule_engine"]
        if self.classifier_ok:
            active.append("ml_classifier")
        if self.embedding_ok:
            active.append("embedding_matcher")

        logger.info(f"RiskEngine ready — active layers: {active}")

    # ─────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────

    def analyze(self, text: str) -> RiskResult:
        """
        Run full detection pipeline on input text.

        Args:
            text: Raw input text from user

        Returns:
            RiskResult with decision, risk_score, explanation,
            and per-layer breakdown
        """
        if not text or not isinstance(text, str):
            return self._empty_result(text or "")

        # ── 0. Preprocess ─────────────────────────────
        processed = self._preprocess(text)

        # ── 1. Rule Engine ────────────────────────────
        rule_result = self.rule_engine.check(processed)

        # Hard override 1 — Critical severity → immediate BLOCK
        if rule_result.max_severity >= SEVERITY_CRITICAL:
            return self._build_result(
                original  = text,
                processed = processed,
                rule_r    = rule_result,
                ml_r      = None,
                emb_r     = None,
                override  = "CRITICAL_RULE",
                fast_path = True
            )

        # ── 2. ML Classifier ──────────────────────────
        ml_result = None
        if self.classifier_ok:
            try:
                ml_result = self.classifier.predict(processed)
            except Exception as e:
                logger.warning(f"ML classifier inference failed: {e}")

        # Hard override 2 — High rule severity + high ML score
        if (
            rule_result.max_severity >= SEVERITY_HIGH
            and ml_result
            and ml_result.ml_score > 0.70
        ):
            return self._build_result(
                original  = text,
                processed = processed,
                rule_r    = rule_result,
                ml_r      = ml_result,
                emb_r     = None,
                override  = "HIGH_RULE_ML_CONFIRM",
                fast_path = True
            )

        # ── 3. Embedding Matcher ───────────────────────
        emb_result = None
        if self.embedding_ok:
            try:
                emb_result = self.embedding.match(processed)
            except Exception as e:
                logger.warning(f"Embedding matcher inference failed: {e}")

        # ── 4. Aggregate and decide ───────────────────
        return self._build_result(
            original  = text,
            processed = processed,
            rule_r    = rule_result,
            ml_r      = ml_result,
            emb_r     = emb_result,
            override  = "",
            fast_path = False
        )

    # ─────────────────────────────────────────
    # SCORING & DECISION
    # ─────────────────────────────────────────

    def _build_result(
        self,
        original:  str,
        processed: str,
        rule_r:    Optional[RuleEngineResult],
        ml_r:      Optional[ClassifierResult],
        emb_r:     Optional[EmbeddingResult],
        override:  str,
        fast_path: bool
    ) -> RiskResult:
        """Compute weighted risk score and build final RiskResult."""

        # Extract scores
        rule_score = rule_r.rule_score      if rule_r  else 0.0
        ml_score   = ml_r.ml_score          if ml_r    else 0.0
        emb_score  = emb_r.embedding_score  if emb_r   else 0.0

        # Adjust weights when layers are unavailable
        if not self.classifier_ok and not self.embedding_ok:
            # Rule only
            risk_score = rule_score
        elif not self.classifier_ok:
            # Rule + embedding
            w_rule = WEIGHT_RULE + WEIGHT_ML * 0.5
            w_emb  = WEIGHT_EMBEDDING + WEIGHT_ML * 0.5
            risk_score = rule_score * w_rule + emb_score * w_emb
        elif not self.embedding_ok:
            # Rule + ML
            w_rule = WEIGHT_RULE + WEIGHT_EMBEDDING * 0.5
            w_ml   = WEIGHT_ML   + WEIGHT_EMBEDDING * 0.5
            risk_score = rule_score * w_rule + ml_score * w_ml
        else:
            # All three layers active
            risk_score = (
                rule_score * WEIGHT_RULE
                + ml_score * WEIGHT_ML
                + emb_score * WEIGHT_EMBEDDING
            )

        risk_score = float(min(1.0, max(0.0, risk_score)))

        # Hard override 3 — any single layer extremely confident
        if rule_score >= 0.95 or ml_score >= 0.95 or emb_score >= 0.95:
            risk_score = max(risk_score, 0.90)
            if not override:
                override = "SINGLE_LAYER_HIGH_CONFIDENCE"

        # Fast path override scores are always high
        if fast_path and override == "CRITICAL_RULE":
            risk_score = max(risk_score, 0.95)
        elif fast_path and override == "HIGH_RULE_ML_CONFIRM":
            risk_score = max(risk_score, 0.80)

        # Final decision from risk score
        if risk_score >= RISK_BLOCK_THRESHOLD:
            decision = DECISION_BLOCK
        elif risk_score >= RISK_MONITOR_THRESHOLD:
            decision = DECISION_MONITOR
        else:
            decision = DECISION_ALLOW

        # Attack category — use most confident layer's output
        attack_category = self._resolve_category(rule_r, ml_r, emb_r, decision)

        # Severity from risk score
        severity = self._score_to_severity(risk_score)

        # Human-readable explanation
        explanation = self._build_explanation(
            rule_r, ml_r, emb_r, decision, risk_score, override
        )

        # Layer breakdown for dashboard
        layers = self._build_breakdown(rule_r, ml_r, emb_r)

        is_attack = (decision != DECISION_ALLOW)

        logger.debug(
            f"RiskEngine: score={risk_score:.3f} "
            f"decision={decision} "
            f"category={attack_category} "
            f"override={override or 'none'}"
        )

        return RiskResult(
            original_text   = original,
            processed_text  = processed,
            risk_score      = round(risk_score, 4),
            decision        = decision,
            attack_category = attack_category,
            severity        = severity,
            explanation     = explanation,
            is_attack       = is_attack,
            layers          = layers,
            override_applied= override,
            fast_path       = fast_path,
        )

    def _resolve_category(
        self,
        rule_r: Optional[RuleEngineResult],
        ml_r:   Optional[ClassifierResult],
        emb_r:  Optional[EmbeddingResult],
        decision: str
    ) -> str:
        """
        Pick the most meaningful attack category from all layers.

        Priority:
          1. Rule engine categories (deterministic)
          2. ML classifier category (if confident)
          3. Embedding top category
          4. "benign" if decision is ALLOW
        """
        if decision == DECISION_ALLOW:
            return "benign"

        # Rule engine categories — most reliable
        if rule_r and rule_r.categories_hit:
            return rule_r.categories_hit[0]

        # ML classifier — if it detected an attack
        if ml_r and ml_r.is_attack:
            return ml_r.predicted_category

        # Embedding — fallback
        if emb_r and emb_r.is_attack:
            return emb_r.top_category

        return "direct_jailbreak"  # safe fallback

    def _score_to_severity(self, risk_score: float) -> int:
        """Map 0.0–1.0 risk score to 1–5 severity tier."""
        if risk_score >= 0.85:
            return SEVERITY_CRITICAL
        elif risk_score >= 0.65:
            return SEVERITY_HIGH
        elif risk_score >= 0.40:
            return SEVERITY_MEDIUM
        elif risk_score >= 0.20:
            return SEVERITY_LOW
        else:
            return SEVERITY_MINIMAL

    def _build_explanation(
        self,
        rule_r:    Optional[RuleEngineResult],
        ml_r:      Optional[ClassifierResult],
        emb_r:     Optional[EmbeddingResult],
        decision:  str,
        score:     float,
        override:  str
    ) -> str:
        """Build a concise human-readable explanation."""
        parts = []

        if override == "CRITICAL_RULE":
            rule_cats = ", ".join(rule_r.categories_hit) if rule_r else "unknown"
            return (
                f"CRITICAL rule match: {rule_cats}. "
                f"Severity 5 — immediate block."
            )

        if override == "HIGH_RULE_ML_CONFIRM":
            return (
                f"High-severity rule match confirmed by ML classifier "
                f"(ml_score={ml_r.ml_score:.2f}). Blocked."
            )

        if rule_r and rule_r.matches:
            parts.append(
                f"Rule engine: {len(rule_r.matches)} match(es), "
                f"severity={rule_r.max_severity}"
            )

        if ml_r and ml_r.is_attack:
            parts.append(
                f"ML: {ml_r.predicted_category} "
                f"(score={ml_r.ml_score:.2f})"
            )

        if emb_r and emb_r.is_attack:
            parts.append(
                f"Embedding: similar to {emb_r.top_category} "
                f"(sim={emb_r.top_similarity:.2f})"
            )

        if not parts:
            if decision == DECISION_ALLOW:
                return "No attack signals detected across all layers."
            else:
                return f"Aggregated risk score {score:.2f} exceeds threshold."

        return " | ".join(parts) + f" | risk={score:.2f}"

    def _build_breakdown(
        self,
        rule_r: Optional[RuleEngineResult],
        ml_r:   Optional[ClassifierResult],
        emb_r:  Optional[EmbeddingResult]
    ) -> LayerBreakdown:
        """Build LayerBreakdown dataclass from layer results."""
        bd = LayerBreakdown()

        if rule_r:
            bd.rule_score      = rule_r.rule_score
            bd.rule_decision   = rule_r.decision
            bd.rule_severity   = rule_r.max_severity
            bd.rule_categories = rule_r.categories_hit
            bd.rule_matches    = len(rule_r.matches)

        if ml_r:
            bd.ml_score      = ml_r.ml_score
            bd.ml_decision   = DECISION_BLOCK if ml_r.is_attack else DECISION_ALLOW
            bd.ml_category   = ml_r.predicted_category
            bd.ml_confidence = ml_r.confidence

        if emb_r:
            bd.embedding_score     = emb_r.embedding_score
            bd.embedding_decision  = emb_r.decision
            bd.embedding_category  = emb_r.top_category
            bd.embedding_sim       = emb_r.top_similarity
            bd.embedding_available = emb_r.index_available

        return bd

    # ─────────────────────────────────────────
    # PREPROCESSING
    # ─────────────────────────────────────────

    def _preprocess(self, text: str) -> str:
        """
        Run text through decoder + normalizer before detection.
        This catches encoding attacks before any rule/ML processing.
        """
        try:
            decoded_result    = self.decoder.run_all(text)
            decoded_text      = decoded_result.get("most_likely_decoded", text) or text
            normalized_result = self.normalizer.process(decoded_text)
            return normalized_result.get("text_for_detection", decoded_text) or text
        except Exception as e:
            logger.warning(f"Preprocessing failed, using raw text: {e}")
            return text

    # ─────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────

    def _empty_result(self, text: str) -> RiskResult:
        """Return a safe ALLOW result for empty/invalid input."""
        return RiskResult(
            original_text   = text,
            processed_text  = text,
            risk_score      = 0.0,
            decision        = DECISION_ALLOW,
            attack_category = "benign",
            severity        = SEVERITY_MINIMAL,
            explanation     = "Empty or invalid input.",
            is_attack       = False,
        )

    def status(self) -> dict:
        """Return current layer availability status."""
        return {
            "rule_engine":  True,
            "ml_classifier": self.classifier_ok,
            "embedding":     self.embedding_ok,
            "layers_active": sum([
                True,
                self.classifier_ok,
                self.embedding_ok
            ])
        }

    def analyze_batch(self, texts: list) -> list:
        """
        Analyze a list of texts.
        Returns list of RiskResult in same order as input.
        """
        return [self.analyze(t) for t in texts]