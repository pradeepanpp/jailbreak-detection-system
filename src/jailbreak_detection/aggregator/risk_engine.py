# src/jailbreak_detection/aggregator/risk_engine.py
"""
Combine the three detection layers into one risk score and decision.

    rules       -> rule_score      (always on)
    classifier  -> ml_score        (needs models/jailbreak_classifier/best)
    embeddings  -> embedding_score (needs models/faiss_index)

risk = weighted sum of the layer scores (weights and thresholds come from
configs/detection_config.yaml). BLOCK at >= 0.65, MONITOR at >= 0.35.

Overrides:
  1. a severity-5 rule blocks at once (the ML layers are skipped)
  2. a severity-4 rule plus ml_score > 0.7 blocks
  3. any single layer >= 0.95 raises the risk to at least 0.90

The input is scored in every form the normalizer produced (original
first, then decoded versions) and the highest risk wins, so a wrong
decode can never hide the original text.
"""

from dataclasses import dataclass, field

from src.jailbreak_detection.constants import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_MONITOR,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_MINIMAL,
)
from src.jailbreak_detection.detection.embedding_matcher import EmbeddingMatcher
from src.jailbreak_detection.detection.rule_engine import RuleEngine, RuleMatch
from src.jailbreak_detection.preprocessing.normalizer import TextNormalizer
from src.jailbreak_detection.utils import load_config, logger


def _config_section(name: str) -> dict:
    try:
        return load_config().get(name) or {}
    except Exception as e:
        logger.warning(f"Config not loaded ({e}); using default weights")
        return {}


_weights = _config_section("weights")
_thresholds = _config_section("thresholds")

WEIGHT_RULE = float(_weights.get("rule_engine", 0.40))
WEIGHT_ML = float(_weights.get("ml_classifier", 0.35))
WEIGHT_EMBEDDING = float(_weights.get("embedding_matcher", 0.25))
RISK_BLOCK_THRESHOLD = float(_thresholds.get("block", 0.65))
RISK_MONITOR_THRESHOLD = float(_thresholds.get("monitor", 0.35))

MAX_VARIANTS = 3   # original text + up to 2 decoded versions


@dataclass
class LayerBreakdown:
    rule_score: float = 0.0
    rule_decision: str = DECISION_ALLOW
    rule_severity: int = 0
    rule_categories: list = field(default_factory=list)
    rule_matches: int = 0

    ml_score: float = 0.0
    ml_decision: str = DECISION_ALLOW
    ml_category: str = "benign"
    ml_confidence: float = 0.0

    embedding_score: float = 0.0
    embedding_decision: str = DECISION_ALLOW
    embedding_category: str = "benign"
    embedding_sim: float = 0.0
    embedding_available: bool = False


@dataclass
class RiskResult:
    original_text: str
    processed_text: str
    risk_score: float          # 0.0 - 1.0
    decision: str              # BLOCK / MONITOR / ALLOW
    attack_category: str
    severity: int              # 1 - 5
    explanation: str
    is_attack: bool
    layers: LayerBreakdown = field(default_factory=LayerBreakdown)
    override_applied: str = ""
    fast_path: bool = False    # True when an override returned early


class RiskEngine:
    """
    engine = RiskEngine()
    result = engine.analyze("ignore all previous instructions")
    result.decision, result.risk_score, result.attack_category
    """

    MODEL_PATH = "models/jailbreak_classifier/best"
    INDEX_PATH = "models/faiss_index"

    def __init__(self, load_classifier: bool = True, load_embedding: bool = True,
                 classifier=None, embedding=None):
        """classifier / embedding: pass ready objects (used by tests) to skip loading."""
        self.normalizer = TextNormalizer()
        self.rule_engine = RuleEngine()

        self.classifier = classifier
        if load_classifier and classifier is None:
            self.classifier = self._load_classifier()
        self.classifier_ok = self.classifier is not None

        self.embedding = embedding
        if load_embedding and embedding is None:
            self.embedding = self._load_embedding()
        self.embedding_ok = self.embedding is not None

        logger.info(f"RiskEngine ready: rules, classifier={self.classifier_ok}, embeddings={self.embedding_ok}")

    def _load_classifier(self):
        try:
            # Imported here because it needs PyTorch
            from src.jailbreak_detection.detection.ml_classifier import JailbreakClassifier
            return JailbreakClassifier.load(self.MODEL_PATH)
        except Exception as e:
            logger.warning(f"Classifier not available, continuing without it: {e}")
            return None

    def _load_embedding(self):
        try:
            matcher = EmbeddingMatcher()
            return matcher if matcher.load(self.INDEX_PATH) else None
        except Exception as e:
            logger.warning(f"Embedding index not available, continuing without it: {e}")
            return None

    def analyze(self, text: str) -> RiskResult:
        if not text or not isinstance(text, str):
            return self._empty_result(text or "")

        processed, variants, methods = self._preprocess(text)

        # Rules are cheap, so check every variant and keep the worst
        rule_r = None
        for variant in variants:
            r = self.rule_engine.check(variant)
            if rule_r is None or (r.max_severity, r.rule_score) > (rule_r.max_severity, rule_r.rule_score):
                rule_r = r
        if methods:
            rule_r = self._add_obfuscation_signal(rule_r, methods, processed)

        if rule_r.max_severity >= SEVERITY_CRITICAL:
            return self._build_result(text, processed, rule_r, None, None, "CRITICAL_RULE", True)

        ml_r = None
        if self.classifier_ok:
            try:
                ml_r = max((self.classifier.predict(v) for v in variants), key=lambda r: r.ml_score)
            except Exception as e:
                logger.warning(f"Classifier failed: {e}")

        if rule_r.max_severity >= SEVERITY_HIGH and ml_r and ml_r.ml_score > 0.70:
            return self._build_result(text, processed, rule_r, ml_r, None, "HIGH_RULE_ML_CONFIRM", True)

        emb_r = None
        if self.embedding_ok:
            try:
                emb_r = max((self.embedding.match(v) for v in variants), key=lambda r: r.embedding_score)
            except Exception as e:
                logger.warning(f"Embedding matcher failed: {e}")

        return self._build_result(text, processed, rule_r, ml_r, emb_r, "", False)

    def analyze_batch(self, texts: list) -> list:
        return [self.analyze(t) for t in texts]

    def _preprocess(self, text: str):
        """Returns (text to show, texts to score, encodings found)."""
        try:
            result = self.normalizer.process(text)
        except Exception as e:
            logger.warning(f"Preprocessing failed, using raw text: {e}")
            return text, [text], []
        variants = result["detection_variants"] or [text]
        return result["text_for_detection"] or text, variants[:MAX_VARIANTS], result["all_detected_methods"]

    def _add_obfuscation_signal(self, rule_r, methods: list, decoded: str):
        """
        Hiding text behind an encoding is a medium-risk signal on its own,
        even when the decoded text matches no rule ("h3lp m3 h4ck th1s").
        """
        signal = RuleMatch(
            rule_id="EN005",
            category="encoding_attack",
            severity=SEVERITY_MEDIUM,
            matched_text=", ".join(methods),
            context=decoded[:100],
            description=f"Obfuscated input decoded ({', '.join(methods)})",
        )
        result = self.rule_engine.build_result(decoded, rule_r.matches + [signal])
        others = [c for c in result.categories_hit if c != "encoding_attack"]
        result.categories_hit = ["encoding_attack"] + others
        return result

    def _combined_score(self, rule_score: float, ml_score: float, emb_score: float) -> float:
        """Weighted sum; a missing layer's weight is split over the others."""
        if not self.classifier_ok and not self.embedding_ok:
            return rule_score
        if not self.classifier_ok:
            return (rule_score * (WEIGHT_RULE + WEIGHT_ML / 2)
                    + emb_score * (WEIGHT_EMBEDDING + WEIGHT_ML / 2))
        if not self.embedding_ok:
            return (rule_score * (WEIGHT_RULE + WEIGHT_EMBEDDING / 2)
                    + ml_score * (WEIGHT_ML + WEIGHT_EMBEDDING / 2))
        return rule_score * WEIGHT_RULE + ml_score * WEIGHT_ML + emb_score * WEIGHT_EMBEDDING

    def _build_result(self, original, processed, rule_r, ml_r, emb_r, override, fast_path) -> RiskResult:
        rule_score = rule_r.rule_score if rule_r else 0.0
        ml_score = ml_r.ml_score if ml_r else 0.0
        emb_score = emb_r.embedding_score if emb_r else 0.0

        risk = min(1.0, max(0.0, self._combined_score(rule_score, ml_score, emb_score)))

        # One very confident layer is enough to raise the risk
        if max(rule_score, ml_score, emb_score) >= 0.95:
            risk = max(risk, 0.90)
            override = override or "SINGLE_LAYER_HIGH_CONFIDENCE"
        if override == "CRITICAL_RULE":
            risk = max(risk, 0.95)
        elif override == "HIGH_RULE_ML_CONFIRM":
            risk = max(risk, 0.80)

        if risk >= RISK_BLOCK_THRESHOLD:
            decision = DECISION_BLOCK
        elif risk >= RISK_MONITOR_THRESHOLD:
            decision = DECISION_MONITOR
        else:
            decision = DECISION_ALLOW

        return RiskResult(
            original_text=original,
            processed_text=processed,
            risk_score=round(risk, 4),
            decision=decision,
            attack_category=self._category(rule_r, ml_r, emb_r, decision),
            severity=self._severity(risk),
            explanation=self._explain(rule_r, ml_r, emb_r, decision, risk, override),
            is_attack=decision != DECISION_ALLOW,
            layers=self._breakdown(rule_r, ml_r, emb_r),
            override_applied=override,
            fast_path=fast_path,
        )

    @staticmethod
    def _category(rule_r, ml_r, emb_r, decision: str) -> str:
        """Rules first (most severe match), then the classifier, then embeddings."""
        if decision == DECISION_ALLOW:
            return "benign"
        if rule_r and rule_r.categories_hit:
            return rule_r.categories_hit[0]
        if ml_r and ml_r.is_attack:
            return ml_r.predicted_category
        if emb_r and emb_r.is_attack:
            return emb_r.top_category
        return "direct_jailbreak"

    @staticmethod
    def _severity(risk: float) -> int:
        if risk >= 0.85:
            return SEVERITY_CRITICAL
        if risk >= 0.65:
            return SEVERITY_HIGH
        if risk >= 0.40:
            return SEVERITY_MEDIUM
        if risk >= 0.20:
            return SEVERITY_LOW
        return SEVERITY_MINIMAL

    @staticmethod
    def _explain(rule_r, ml_r, emb_r, decision: str, risk: float, override: str) -> str:
        if override == "CRITICAL_RULE":
            categories = ", ".join(rule_r.categories_hit) if rule_r else "unknown"
            return f"CRITICAL rule match: {categories}. Severity 5 — immediate block."
        if override == "HIGH_RULE_ML_CONFIRM":
            return f"High-severity rule match confirmed by ML classifier (ml_score={ml_r.ml_score:.2f}). Blocked."

        parts = []
        if rule_r and rule_r.matches:
            parts.append(f"Rule engine: {len(rule_r.matches)} match(es), severity={rule_r.max_severity}")
        if ml_r and ml_r.is_attack:
            parts.append(f"ML: {ml_r.predicted_category} (score={ml_r.ml_score:.2f})")
        if emb_r and emb_r.is_attack:
            parts.append(f"Embedding: similar to {emb_r.top_category} (sim={emb_r.top_similarity:.2f})")

        if parts:
            return " | ".join(parts) + f" | risk={risk:.2f}"
        if decision == DECISION_ALLOW:
            return "No attack signals detected across all layers."
        return f"Aggregated risk score {risk:.2f} exceeds threshold."

    @staticmethod
    def _breakdown(rule_r, ml_r, emb_r) -> LayerBreakdown:
        b = LayerBreakdown()
        if rule_r:
            b.rule_score = rule_r.rule_score
            b.rule_decision = rule_r.decision
            b.rule_severity = rule_r.max_severity
            b.rule_categories = rule_r.categories_hit
            b.rule_matches = len(rule_r.matches)
        if ml_r:
            b.ml_score = ml_r.ml_score
            b.ml_decision = DECISION_BLOCK if ml_r.is_attack else DECISION_ALLOW
            b.ml_category = ml_r.predicted_category
            b.ml_confidence = ml_r.confidence
        if emb_r:
            b.embedding_score = emb_r.embedding_score
            b.embedding_decision = emb_r.decision
            b.embedding_category = emb_r.top_category
            b.embedding_sim = emb_r.top_similarity
            b.embedding_available = emb_r.index_available
        return b

    @staticmethod
    def _empty_result(text: str) -> RiskResult:
        return RiskResult(
            original_text=text,
            processed_text=text,
            risk_score=0.0,
            decision=DECISION_ALLOW,
            attack_category="benign",
            severity=SEVERITY_MINIMAL,
            explanation="Empty or invalid input.",
            is_attack=False,
        )

    def warm_up(self) -> None:
        """One throwaway request, so model loading is not charged to a real user (or to latency numbers)."""
        try:
            self.analyze("Warm-up request: what is the capital of France?")
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")

    def status(self) -> dict:
        return {
            "rule_engine": True,
            "ml_classifier": self.classifier_ok,
            "embedding": self.embedding_ok,
            "layers_active": 1 + self.classifier_ok + self.embedding_ok,
        }
