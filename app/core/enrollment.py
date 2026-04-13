"""
enrollment.py
-------------
Dataset Enrollment module.

Responsibilities:
    - Walk through Dataset/<person>/ folders.
    - Detect & crop faces using MTCNN.
    - Generate embeddings using FaceNet.
    - Build and persist a FAISS index + JSON metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from collections import defaultdict

import cv2
import numpy as np

from app.config import DATASET_DIR, IMAGE_SIZE
from app.core.detector import FaceDetector
from app.core.embedder import FaceEmbedder
from app.core.recognizer import FaceRecognizer

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}

# Minimum number of successfully extracted face embeddings required per person
# before a centroid is computed.  With fewer samples the centroid is unreliable.
MIN_IMAGES_REQUIRED = 10


class EnrollmentManager:
    """
    Handles reading the dataset, extracting embeddings, and building the
    FAISS recognition index.

    Parameters
    ----------
    dataset_dir : Path
        Root dataset directory containing one sub-folder per person.
    detector    : FaceDetector
    embedder    : FaceEmbedder
    recognizer  : FaceRecognizer
    """

    def __init__(
        self,
        dataset_dir: Path = DATASET_DIR,
        detector:   FaceDetector   = None,
        embedder:   FaceEmbedder   = None,
        recognizer: FaceRecognizer = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.detector    = detector   or FaceDetector()
        self.embedder    = embedder   or FaceEmbedder()
        self.recognizer  = recognizer or FaceRecognizer()

    # ─────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────

    def enroll_all(self) -> Dict:
        """
        Process entire dataset, build FAISS index, and save to disk.

        Strategy: CENTROID-BASED ENROLLMENT
        ------------------------------------
        For each person we collect all per-image embeddings, compute their
        L2-normalised mean (centroid), and store ONLY that centroid in the
        FAISS index.  The centroid is the best single-vector summary of a
        person's face across different poses/lighting because it sits in the
        centre of all their samples in embedding space.
        This approach is far more robust to unseen poses than storing every
        individual snapshot separately.

        Returns
        -------
        dict with keys: persons, total_persons, total_images, total_embeddings
        """
        # Collect raw per-image embeddings grouped by person
        per_person_embs: Dict[str, List[np.ndarray]] = defaultdict(list)
        stats: Dict[str, int] = {}

        person_dirs = sorted(
            [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        )

        if not person_dirs:
            raise FileNotFoundError(
                f"No person sub-folders found in {self.dataset_dir}"
            )

        logger.info("Starting enrollment for %d persons …", len(person_dirs))

        for person_dir in person_dirs:
            name = person_dir.name
            imgs = self._load_images(person_dir)

            if not imgs:
                logger.warning("  [%s] No valid images found – skipping.", name)
                continue

            person_embeddings = self._process_person(name, imgs)

            if not person_embeddings:
                logger.warning("  [%s] No faces detected in any image – skipping.", name)
                continue

            per_person_embs[name].extend(person_embeddings)
            stats[name] = len(person_embeddings)

            logger.info(
                "  [%s] collected %d face embedding(s) from %d image(s).",
                name, len(person_embeddings), len(imgs),
            )

        if not per_person_embs:
            raise RuntimeError("No embeddings generated. Check your dataset.")

        # ── Warn when a person's sample count is below the recommended minimum ──
        for name, embs in per_person_embs.items():
            if len(embs) < MIN_IMAGES_REQUIRED:
                logger.warning(
                    "  [%s] Only %d embedding(s) collected — centroid may be "
                    "unreliable. Recommend at least %d varied images per person.",
                    name, len(embs), MIN_IMAGES_REQUIRED,
                )

        # ── Build centroids: one L2-normalised mean vector per person ──────
        centroid_embeddings: List[np.ndarray] = []
        centroid_labels:     List[str]        = []

        for name, embs in per_person_embs.items():
            stacked  = np.vstack(embs)              # (K, 512)
            centroid = stacked.mean(axis=0)         # (512,)  — mean across all samples
            norm     = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm          # re-normalise to unit length
            centroid_embeddings.append(centroid)
            centroid_labels.append(name)
            logger.info(
                "  [%s] centroid built from %d embedding(s).",
                name, len(embs),
            )

        all_centroids = np.vstack(centroid_embeddings).astype(np.float32)  # (P, 512)
        self.recognizer.build_index(all_centroids, centroid_labels)
        self.recognizer.save()

        total_images = sum(stats.values())
        return {
            "persons":          list(per_person_embs.keys()),
            "total_persons":    len(per_person_embs),
            "total_images":     total_images,
            "total_embeddings": len(centroid_labels),  # = number of persons
            "per_person":       stats,
        }

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _load_images(self, folder: Path) -> List[np.ndarray]:
        """Load all supported images from a folder as BGR ndarrays."""
        images = []
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("    Could not read image: %s", img_path)
                continue
            images.append(img)
        return images

    def _process_person(
        self,
        name: str,
        images: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Detect face(s) in each image, generate embeddings.

        IMPORTANT: Uses the same detector.detect() → embedder.get_embedding()
        pipeline as the recognition endpoint to ensure consistent preprocessing.
        Using a different pipeline (e.g. manual BGR crop) produces different
        embeddings for the same face, causing low similarity scores.

        Returns list of (512,) embedding vectors.
        """
        embeddings = []
        for img in images:
            # Use the same MTCNN tensor pipeline as recognize.py
            faces_tensor, _ = self.detector.detect(img)
            if faces_tensor is None:
                logger.debug("  [%s] No face in one image – trying fallback crop.", name)
                # Fallback: manual crop for images where MTCNN tensor path fails
                crops = self.detector.detect_and_crop(img)
                for crop in crops:
                    emb = self.embedder.get_embedding_from_numpy(crop)
                    embeddings.append(emb)
                continue

            # faces_tensor shape: (N, 3, 160, 160) — already normalized by MTCNN
            batch_embs = self.embedder.get_embedding(faces_tensor)  # (N, 512)
            for emb in batch_embs:
                embeddings.append(emb)
        return embeddings
