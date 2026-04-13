"""
detector.py
-----------
Face Detection module using MTCNN (Multi-Task Cascaded Convolutional Networks).

Responsibilities:
    - Detect face bounding boxes and landmarks in an image/frame.
    - Crop and preprocess detected faces (resize to 160×160 for FaceNet).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from app.config import (
    IMAGE_SIZE,
    MIN_FACE_SIZE,
    MTCNN_THRESHOLDS,
    MTCNN_MIN_MARGIN,
)

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Wrapper around facenet_pytorch.MTCNN for robust face detection.

    Parameters
    ----------
    device : str | None
        Torch device string ('cuda' or 'cpu'). Auto-detected if None.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("FaceDetector initialising on device: %s", self.device)

        self.mtcnn = MTCNN(
            image_size=IMAGE_SIZE,
            margin=MTCNN_MIN_MARGIN,
            min_face_size=20,            # lowered from 40 so small faces in compressed images are found
            thresholds=[0.5, 0.6, 0.6],  # relaxed from [0.6, 0.7, 0.7] for better recall
            factor=0.709,
            post_process=True,       # Normalise to [-1, 1] for FaceNet
            keep_all=True,           # Return ALL faces in the frame
            device=self.device,
        )
        logger.info("MTCNN loaded successfully.")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        """
        Detect faces in an BGR numpy image (OpenCV format).

        Returns
        -------
        faces : torch.Tensor | None
            Tensor of shape (N, 3, 160, 160) with post-processed face crops,
            or None if no faces are detected.
        boxes : np.ndarray | None
            Array of shape (N, 4) with [x1, y1, x2, y2] bounding boxes,
            or None if no faces detected.
        """
        rgb = self._bgr_to_rgb(image)
        pil_img = Image.fromarray(rgb)

        try:
            # Step 1: Get bounding boxes + probs (always returns 2 values)
            boxes, probs = self.mtcnn.detect(pil_img)
        except Exception as exc:
            logger.warning("MTCNN detect() failed: %s", exc)
            return None, None

        if boxes is None or probs is None:
            logger.debug("No faces detected in image.")
            return None, None

        # Filter by confidence threshold
        valid_indices = [
            i for i, p in enumerate(probs)
            if p is not None and p > 0.80
        ]
        if not valid_indices:
            logger.debug("All detections below confidence threshold.")
            return None, None

        try:
            # Step 2: Get face tensors (post-processed for FaceNet input)
            # mtcnn() with return_prob=True returns (faces_tensor, probs_array) — 2 values
            faces, _ = self.mtcnn(pil_img, return_prob=True)
        except Exception as exc:
            logger.warning("MTCNN face extraction failed: %s", exc)
            return None, None

        if faces is None:
            return None, None

        # Ensure faces is a 4-D tensor (N, C, H, W)
        if faces.ndim == 3:
            faces = faces.unsqueeze(0)

        valid_mask = torch.zeros(len(probs), dtype=torch.bool)
        for i in valid_indices:
            valid_mask[i] = True

        # Guard: valid_mask must match faces batch size
        n_faces = faces.shape[0]
        if len(valid_mask) != n_faces:
            # Fallback: take all faces
            logger.debug("Mask/faces size mismatch – returning all faces.")
            boxes_np = np.array(boxes, dtype=float)
            return faces, boxes_np.astype(int)

        faces = faces[valid_mask]
        boxes_np = np.array(boxes)[valid_mask.numpy()]

        return faces, boxes_np.astype(int)

    def detect_and_crop(
        self,
        image: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Detect faces and return cropped BGR images resized to IMAGE_SIZE.
        Used during dataset enrollment where we need raw pixel crops.

        Returns
        -------
        List of BGR crops (may be empty).
        """
        rgb = self._bgr_to_rgb(image)
        pil_img = Image.fromarray(rgb)

        try:
            boxes, probs = self.mtcnn.detect(pil_img)
        except Exception as exc:
            logger.warning("Detection error: %s", exc)
            return []

        if boxes is None:
            return []

        crops: List[np.ndarray] = []
        for box, prob in zip(boxes, probs):
            if prob < 0.80:  # lowered from 0.90 for better recall on smaller images
                continue
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
            crops.append(crop_resized)

        return crops

    # ─────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert OpenCV BGR image to RGB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
