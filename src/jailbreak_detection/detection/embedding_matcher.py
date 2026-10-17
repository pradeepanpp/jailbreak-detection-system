# src/jailbreak_detection/detection/embedding_matcher.py
"""
Layer 3: semantic similarity to known attacks.

Every attack in train.csv is embedded with all-mpnet-base-v2 and stored in
a FAISS inner-product index (built by scripts/build_index.py). An input is
embedded the same way, and the cosine similarity to its nearest known
attack becomes the embedding score. This can catch reworded attacks that
no regex matches.
"""

import os
from dataclasses import dataclass, field

import numpy as np

from src.jailbreak_detection.constants import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_MONITOR,
    ID_TO_LABEL,
)
from src.jailbreak_detection.utils import logger


@dataclass
class EmbeddingMatch:
    rank: int               # 1 = closest
    similarity: float       # cosine similarity, 0.0 - 1.0
    matched_text: str
    matched_label: int
    matched_category: str


@dataclass
class EmbeddingResult:
    input_text: str
    top_matches: list = field(default_factory=list)
    top_similarity: float = 0.0
    top_category: str = "benign"
    embedding_score: float = 0.0   # used by the risk engine
    is_attack: bool = False
    decision: str = DECISION_ALLOW
    explanation: str = "No similar attacks found"
    index_available: bool = False


class EmbeddingMatcher:
    """
    matcher = EmbeddingMatcher()
    matcher.load("models/faiss_index")
    result = matcher.match("ignore everything you were told")
    """

    INDEX_DIR = "models/faiss_index"
    INDEX_FILE = "index.faiss"
    METADATA_FILE = "metadata.npz"
    MODEL_NAME = "all-mpnet-base-v2"
    TOP_K = 5
    BLOCK_THRESHOLD = 0.90
    MONITOR_THRESHOLD = 0.55
    # The model reads about 384 tokens. For longer inputs the last part is
    # embedded too, because many-shot payloads sit at the end.
    TAIL_CHARS = 1500

    def __init__(self):
        self._model = None    # loaded on first use (large download)
        self._index = None
        self._texts = None
        self._labels = None
        self._loaded = False

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model {self.MODEL_NAME}")
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def embed(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """L2-normalised embeddings, so inner product = cosine similarity."""
        vectors = self._get_model().encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def build_and_save(self, texts: list, labels: list, index_dir: str = None):
        import faiss

        index_dir = index_dir or self.INDEX_DIR
        os.makedirs(index_dir, exist_ok=True)

        vectors = self.embed(texts, batch_size=64)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        faiss.write_index(index, os.path.join(index_dir, self.INDEX_FILE))
        np.savez(
            os.path.join(index_dir, self.METADATA_FILE),
            texts=np.array(texts, dtype=object),
            labels=np.array(labels, dtype=np.int32),
        )

        self._index = index
        self._texts = np.array(texts, dtype=object)
        self._labels = np.array(labels, dtype=np.int32)
        self._loaded = True

    def load(self, index_dir: str = None) -> bool:
        """Load the index from disk. Returns False if it has not been built."""
        import faiss

        index_dir = index_dir or self.INDEX_DIR
        index_path = os.path.join(index_dir, self.INDEX_FILE)
        meta_path = os.path.join(index_dir, self.METADATA_FILE)
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            logger.warning(f"No FAISS index in {index_dir}. Run scripts/build_index.py")
            return False

        self._index = faiss.read_index(index_path)
        meta = np.load(meta_path, allow_pickle=True)
        self._texts = meta["texts"]
        self._labels = meta["labels"]
        self._loaded = True
        logger.info(f"FAISS index loaded: {self._index.ntotal} vectors")
        return True

    def match(self, text: str) -> EmbeddingResult:
        if not text or not isinstance(text, str):
            return EmbeddingResult(input_text="", index_available=self._loaded)
        if not self._loaded:
            return EmbeddingResult(input_text=text[:100], explanation="Index not built — run build_index.py")

        # Search with the whole text, and also its end if the text is long;
        # keep whichever is closer to a known attack
        queries = [text] if len(text) <= self.TAIL_CHARS else [text, text[-self.TAIL_CHARS:]]
        k = min(self.TOP_K, self._index.ntotal)
        scores, indices = self._index.search(self.embed(queries), k)
        best = int(np.argmax(scores[:, 0]))

        matches = []
        for rank, (score, idx) in enumerate(zip(scores[best], indices[best]), start=1):
            if idx < 0:   # FAISS uses -1 for empty slots
                continue
            label = int(self._labels[idx])
            matches.append(EmbeddingMatch(
                rank=rank,
                similarity=float(np.clip(score, 0.0, 1.0)),
                matched_text=str(self._texts[idx])[:120],
                matched_label=label,
                matched_category=ID_TO_LABEL.get(label, "unknown"),
            ))
        if not matches:
            return EmbeddingResult(input_text=text[:100], explanation="No matches found in index",
                                   index_available=True)

        top = matches[0]
        # Only attacks are indexed, so the nearest neighbour is an attack;
        # its similarity is the score
        score = 0.0 if top.matched_category == "benign" else top.similarity
        if score >= self.BLOCK_THRESHOLD:
            decision = DECISION_BLOCK
            explanation = f"High similarity ({top.similarity:.3f}) to known {top.matched_category} attack"
        elif score >= self.MONITOR_THRESHOLD:
            decision = DECISION_MONITOR
            explanation = f"Moderate similarity ({top.similarity:.3f}) to known {top.matched_category} attack"
        else:
            decision = DECISION_ALLOW
            explanation = f"Low similarity ({top.similarity:.3f}) to nearest known attack"

        return EmbeddingResult(
            input_text=text[:100],
            top_matches=matches,
            top_similarity=top.similarity,
            top_category=top.matched_category,
            embedding_score=score,
            is_attack=decision != DECISION_ALLOW,
            decision=decision,
            explanation=explanation,
            index_available=True,
        )

    def is_ready(self) -> bool:
        return self._loaded

    def index_size(self) -> int:
        return 0 if self._index is None else self._index.ntotal

    def get_category_distribution(self) -> dict:
        if self._labels is None:
            return {}
        return {name: int((self._labels == label_id).sum()) for label_id, name in ID_TO_LABEL.items()}
