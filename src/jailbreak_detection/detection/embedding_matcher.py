# src/jailbreak_detection/detection/embedding_matcher.py
"""
Layer 3 — Embedding Similarity Matcher.

Uses sentence embeddings + FAISS index to catch attacks
that are semantically similar to known attacks — even when
exact wording is completely different.

Why this layer exists:
  Rule engine  → catches exact/pattern matches
  ML Classifier → catches category-level patterns
  Embedding    → catches NOVEL attacks similar to known ones

Example:
  Known attack : "ignore all previous instructions"
  Novel attack : "disregard every directive you received earlier"
  → Rule engine misses it (different words)
  → Embedding catches it (same semantic meaning)

Flow:
  1. build_index.py runs once → builds FAISS index from train.csv
  2. At inference: embed incoming text → search index → return
     top-k matches + similarity scores → aggregator uses these

Model: all-mpnet-base-v2 (best quality sentence embeddings)
Index: FAISS IndexFlatIP (inner product = cosine on normalized vecs)
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from src.jailbreak_detection.utils import logger
from src.jailbreak_detection.constants import (
    LABEL_MAP,
    ID_TO_LABEL,
    DECISION_BLOCK,
    DECISION_MONITOR,
    DECISION_ALLOW,
)


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class EmbeddingMatch:
    """A single nearest-neighbour match from the FAISS index."""
    rank:           int     # 1 = closest match
    similarity:     float   # cosine similarity 0.0 - 1.0
    matched_text:   str     # the matched training sample text
    matched_label:  int     # integer label of matched sample
    matched_category: str   # human-readable category name


@dataclass
class EmbeddingResult:
    """Full result from the embedding matcher."""
    input_text:       str
    top_matches:      List[EmbeddingMatch] = field(default_factory=list)
    top_similarity:   float  = 0.0    # highest cosine similarity found
    top_category:     str    = "benign"
    embedding_score:  float  = 0.0    # 0.0-1.0 used by aggregator
    is_attack:        bool   = False
    decision:         str    = DECISION_ALLOW
    explanation:      str    = "No similar attacks found"
    index_available:  bool   = False  # False if index not built yet


# ─────────────────────────────────────────────
# EMBEDDING MATCHER
# ─────────────────────────────────────────────

class EmbeddingMatcher:
    """
    Semantic similarity detection using FAISS.

    Usage:
        # One-time index build (run scripts/build_index.py)
        matcher = EmbeddingMatcher()
        matcher.build_and_save(texts, labels, "models/faiss_index")

        # Inference
        matcher = EmbeddingMatcher()
        matcher.load("models/faiss_index")
        result = matcher.match("ignore everything you were told")
    """

    INDEX_DIR      = "models/faiss_index"
    INDEX_FILE     = "index.faiss"
    METADATA_FILE  = "metadata.npz"
    MODEL_NAME     = "all-mpnet-base-v2"
    TOP_K          = 5       # number of neighbours to retrieve
    BLOCK_THRESHOLD   = 0.90  # above this → BLOCK
    MONITOR_THRESHOLD = 0.55  # above this → MONITOR

    def __init__(self):
        self._model  = None   # SentenceTransformer — lazy loaded
        self._index  = None   # FAISS index
        self._texts  = None   # np.array of stored texts
        self._labels = None   # np.array of stored labels
        self._loaded = False

        logger.info("EmbeddingMatcher initialized (index not loaded yet)")

    # ─────────────────────────────────────────
    # LAZY MODEL LOADER
    # ─────────────────────────────────────────

    def _get_model(self):
        """Load SentenceTransformer only when first needed."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.MODEL_NAME}")
            self._model = SentenceTransformer(self.MODEL_NAME)
            logger.info("Embedding model loaded")
        return self._model

    # ─────────────────────────────────────────
    # EMBEDDING
    # ─────────────────────────────────────────

    def embed(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of texts into L2-normalized vectors.
        Normalization converts dot-product search to cosine similarity.

        Args:
            texts     : list of strings
            batch_size: processing batch size

        Returns:
            np.ndarray shape (N, embedding_dim), float32, L2-normalized
        """
        model = self._get_model()

        embeddings = model.encode(
            texts,
            batch_size        = batch_size,
            show_progress_bar = len(texts) > 100,
            convert_to_numpy  = True,
            normalize_embeddings = True   # ← cosine similarity via dot product
        )

        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (1, dim)."""
        return self.embed([text])

    # ─────────────────────────────────────────
    # INDEX BUILD
    # ─────────────────────────────────────────

    def build_and_save(
        self,
        texts:  list,
        labels: list,
        index_dir: str = None
    ):
        """
        Build FAISS index from texts+labels and save to disk.
        Called once by scripts/build_index.py.

        Args:
            texts     : list of training text strings
            labels    : list of integer labels (matching LABEL_MAP)
            index_dir : directory to save index files
        """
        import faiss

        index_dir = index_dir or self.INDEX_DIR
        os.makedirs(index_dir, exist_ok=True)

        logger.info(f"Building FAISS index from {len(texts)} samples...")

        # Embed all texts
        embeddings = self.embed(texts, batch_size=64)
        dim        = embeddings.shape[1]

        logger.info(f"Embeddings shape: {embeddings.shape} (dim={dim})")

        # Build inner-product index (cosine on normalized vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        logger.info(f"FAISS index built: {index.ntotal} vectors")

        # Save index
        index_path = os.path.join(index_dir, self.INDEX_FILE)
        faiss.write_index(index, index_path)

        # Save metadata (texts + labels)
        meta_path = os.path.join(index_dir, self.METADATA_FILE)
        np.savez(
            meta_path,
            texts  = np.array(texts,  dtype=object),
            labels = np.array(labels, dtype=np.int32)
        )

        logger.info(f"Index saved  → {index_path}")
        logger.info(f"Metadata saved → {meta_path}")

        # Load into memory for immediate use
        self._index  = index
        self._texts  = np.array(texts, dtype=object)
        self._labels = np.array(labels, dtype=np.int32)
        self._loaded = True

        logger.info("FAISS index ready for inference")

    # ─────────────────────────────────────────
    # INDEX LOAD
    # ─────────────────────────────────────────

    def load(self, index_dir: str = None) -> bool:
        """
        Load FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False if files not found.
        """
        import faiss

        index_dir  = index_dir or self.INDEX_DIR
        index_path = os.path.join(index_dir, self.INDEX_FILE)
        meta_path  = os.path.join(index_dir, self.METADATA_FILE)

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.warning(
                f"FAISS index not found at {index_dir}. "
                f"Run: python scripts/build_index.py"
            )
            return False

        self._index = faiss.read_index(index_path)
        meta        = np.load(meta_path, allow_pickle=True)
        self._texts  = meta["texts"]
        self._labels = meta["labels"]
        self._loaded = True

        logger.info(
            f"FAISS index loaded: {self._index.ntotal} vectors "
            f"from {index_dir}"
        )
        return True

    # ─────────────────────────────────────────
    # INFERENCE
    # ─────────────────────────────────────────

    def match(self, text: str) -> EmbeddingResult:
        """
        Find semantically similar known attacks in the index.

        Args:
            text: input text to check
                  (use text_for_detection from normalizer output)

        Returns:
            EmbeddingResult with similarity scores and decision
        """
        if not text or not isinstance(text, str):
            return EmbeddingResult(
                input_text      = "",
                index_available = self._loaded
            )

        # If index not built yet, return graceful no-op
        if not self._loaded:
            return EmbeddingResult(
                input_text      = text[:100],
                decision        = DECISION_ALLOW,
                explanation     = "Index not built — run build_index.py",
                index_available = False
            )

        # Embed query
        query_vec = self.embed_single(text)   # shape (1, dim)

        # Search top-k
        k          = min(self.TOP_K, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        scores  = scores[0]    # shape (k,)
        indices = indices[0]   # shape (k,)

        # Build match objects
        matches = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0:   # FAISS returns -1 for empty slots
                continue

            label    = int(self._labels[idx])
            category = ID_TO_LABEL.get(label, "unknown")
            sim      = float(np.clip(score, 0.0, 1.0))

            matches.append(EmbeddingMatch(
                rank            = rank,
                similarity      = sim,
                matched_text    = str(self._texts[idx])[:120],
                matched_label   = label,
                matched_category= category
            ))

        if not matches:
            return EmbeddingResult(
                input_text      = text[:100],
                decision        = DECISION_ALLOW,
                explanation     = "No matches found in index",
                index_available = True
            )

        # Top match drives the decision
        top = matches[0]
        top_sim  = top.similarity
        top_cat  = top.matched_category

        # Embedding score: similarity to attack scaled by whether
        # the nearest neighbour is actually an attack
        if top_cat == "benign":
            embedding_score = 0.0
        else:
            embedding_score = top_sim

        # Decision
        if embedding_score >= self.BLOCK_THRESHOLD:
            decision    = DECISION_BLOCK
            explanation = (
                f"High similarity ({top_sim:.3f}) to known "
                f"{top_cat} attack"
            )
        elif embedding_score >= self.MONITOR_THRESHOLD:
            decision    = DECISION_MONITOR
            explanation = (
                f"Moderate similarity ({top_sim:.3f}) to known "
                f"{top_cat} attack"
            )
        else:
            decision    = DECISION_ALLOW
            explanation = (
                f"Low similarity ({top_sim:.3f}) to nearest known attack"
            )

        is_attack = (decision != DECISION_ALLOW)

        logger.debug(
            f"EmbeddingMatcher: top_sim={top_sim:.3f} "
            f"category={top_cat} decision={decision}"
        )

        return EmbeddingResult(
            input_text      = text[:100],
            top_matches     = matches,
            top_similarity  = top_sim,
            top_category    = top_cat,
            embedding_score = embedding_score,
            is_attack       = is_attack,
            decision        = decision,
            explanation     = explanation,
            index_available = True
        )

    # ─────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────

    def is_ready(self) -> bool:
        """Check if index is loaded and ready for inference."""
        return self._loaded

    def index_size(self) -> int:
        """Return number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def get_category_distribution(self) -> dict:
        """Return count of each category in the index."""
        if self._labels is None:
            return {}
        dist = {}
        for label_id, label_name in ID_TO_LABEL.items():
            count = int((self._labels == label_id).sum())
            dist[label_name] = count
        return dist