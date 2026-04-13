"""
recognizer.py
-------------
Face Recognition module using FAISS for fast similarity search.

Responsibilities:
    - Build and persist a FAISS index from enrolled embeddings.
    - Load an existing index from disk.
    - Query the index with a probe embedding and return the nearest match
      with a cosine-similarity confidence score.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.config import (
    COSINE_SIMILARITY_THRESHOLD,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    METADATA_JSON_PATH,
    UNKNOWN_LABEL,
)

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    FAISS-backed cosine-similarity face recogniser.

    The index uses an Inner Product (IP) flat index.  Because all
    embeddings are L2-normalised by the embedder, inner product equals
    cosine similarity, giving us fast, exact nearest-neighbour search.
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.IndexFlatIP] = None
        self.labels: List[str] = []          # parallel to FAISS vectors
        self._label_set: Dict[str, int] = {} # name → count (for stats)

    # ─────────────────────────────────────────
    # Build / Persist
    # ─────────────────────────────────────────

    def build_index(
        self,
        embeddings: np.ndarray,
        labels: List[str],
    ) -> None:
        """
        Build a new FAISS index from a matrix of embeddings.

        Parameters
        ----------
        embeddings : np.ndarray  shape (N, EMBEDDING_DIM)
        labels     : List[str]   length N, parallel to embeddings rows
        """
        assert embeddings.shape[0] == len(labels), "Mismatch between embeddings and labels."

        self.labels = labels
        self._label_set = {}
        for lbl in labels:
            self._label_set[lbl] = self._label_set.get(lbl, 0) + 1

        # Ensure float32 contiguous array for FAISS
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings_f32)

        logger.info(
            "FAISS index built: %d vectors across %d persons.",
            len(labels),
            len(self._label_set),
        )

    def save(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = METADATA_JSON_PATH,
    ) -> None:
        """Persist FAISS index and label metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"labels": self.labels, "label_set": self._label_set}, f, indent=2)

        logger.info("FAISS index saved → %s", index_path)
        logger.info("Metadata saved    → %s", meta_path)

    def load(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = METADATA_JSON_PATH,
    ) -> bool:
        """
        Load FAISS index and metadata from disk.

        Returns True on success, False if files do not exist yet.
        """
        if not index_path.exists() or not meta_path.exists():
            logger.warning("No persisted index found at %s.", index_path)
            return False

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.labels    = meta["labels"]
        self._label_set = meta.get("label_set", {})

        logger.info(
            "FAISS index loaded: %d vectors, %d persons.",
            len(self.labels),
            len(self._label_set),
        )
        return True

    # ─────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────

    def recognize(
        self,
        probe_embedding: np.ndarray,
        threshold: float = COSINE_SIMILARITY_THRESHOLD,
    ) -> Tuple[str, float]:
        """
        Identify the person closest to the probe embedding.

        Parameters
        ----------
        probe_embedding : np.ndarray  shape (512,) or (1, 512)
        threshold       : float  cosine similarity cut-off

        Returns
        -------
        (name, confidence)
            name       : recognised person name or UNKNOWN_LABEL
            confidence : cosine similarity score ∈ [0, 1]
        """
        if self.index is None or self.index.ntotal == 0:
            return UNKNOWN_LABEL, 0.0

        probe = np.ascontiguousarray(
            probe_embedding.reshape(1, -1), dtype=np.float32
        )

        distances, indices = self.index.search(probe, k=1)
        similarity = float(distances[0][0])
        idx        = int(indices[0][0])

        if idx < 0 or similarity < threshold:
            return UNKNOWN_LABEL, similarity

        return self.labels[idx], similarity

    def recognize_batch(
        self,
        probe_embeddings: np.ndarray,
        threshold: float = COSINE_SIMILARITY_THRESHOLD,
    ) -> List[Tuple[str, float]]:
        """
        Recognise a batch of face embeddings at once.

        Parameters
        ----------
        probe_embeddings : np.ndarray  shape (N, 512)

        Returns
        -------
        List of (name, confidence) tuples, length N.
        """
        if self.index is None or self.index.ntotal == 0:
            return [(UNKNOWN_LABEL, 0.0)] * len(probe_embeddings)

        probes = np.ascontiguousarray(probe_embeddings, dtype=np.float32)
        distances, indices = self.index.search(probes, k=1)

        results: List[Tuple[str, float]] = []
        for dist, idx in zip(distances, indices):
            similarity = float(dist[0])
            i          = int(idx[0])
            if i < 0 or similarity < threshold:
                results.append((UNKNOWN_LABEL, similarity))
            else:
                results.append((self.labels[i], similarity))

        return results

    # ─────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the index has been built or loaded."""
        return self.index is not None and self.index.ntotal > 0

    @property
    def enrolled_persons(self) -> List[str]:
        """List of unique enrolled person names."""
        return list(self._label_set.keys())
