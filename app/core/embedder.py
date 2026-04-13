"""
embedder.py
-----------
Face Embedding module using FaceNet (InceptionResnetV1).

Responsibilities:
    - Load the pre-trained FaceNet model (VGGFace2 weights).
    - Accept a face tensor (output from MTCNN) and produce L2-normalised
      512-dimensional embedding vectors.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """
    Wraps the InceptionResnetV1 (FaceNet) model for embedding generation.

    Parameters
    ----------
    device : str | None
        Torch device. Auto-detected if None.
    pretrained : str
        Which pretrained weights to load ('vggface2' or 'casia-webface').
    """

    def __init__(
        self,
        device: Optional[str] = None,
        pretrained: str = "vggface2",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("FaceEmbedder loading on device: %s", self.device)

        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
        logger.info("FaceNet (InceptionResnetV1 / %s) loaded.", pretrained)

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def get_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate embeddings for one or more face crops.

        Parameters
        ----------
        face_tensor : torch.Tensor
            Shape (N, 3, 160, 160) or (3, 160, 160) for a single face.
            Values should be in [-1, 1] (MTCNN post-processed output).

        Returns
        -------
        np.ndarray
            Shape (N, 512) of L2-normalised embedding vectors.
        """
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)

        face_tensor = face_tensor.to(self.device)

        with torch.no_grad():
            embeddings = self.model(face_tensor)  # (N, 512)

        # L2 normalise so cosine similarity = dot product
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def get_embedding_from_numpy(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Convenience method: takes a single BGR face image (H×W×3),
        converts it to the proper tensor, and returns a (512,) embedding.

        Parameters
        ----------
        face_bgr : np.ndarray
            Face crop in BGR format, shape (160, 160, 3).

        Returns
        -------
        np.ndarray  shape (512,)
        """
        import cv2
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # Normalise to [-1, 1]
        tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)
        tensor = (tensor - 127.5) / 128.0
        embedding = self.get_embedding(tensor)  # (1, 512)
        return embedding[0]
