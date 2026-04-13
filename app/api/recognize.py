"""
recognize.py
-------------
POST /recognize
    Accept an uploaded image, detect all faces, return identity + confidence.
    Also marks attendance and logs unknown faces automatically.
"""

from __future__ import annotations

import io
import logging
from typing import List

import cv2
import numpy as np
from PIL import Image as PILImage
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import UNKNOWN_LABEL
from app.core.attendance_manager import AttendanceManager
from app.core.detector import FaceDetector
from app.core.embedder import FaceEmbedder
from app.core.recognizer import FaceRecognizer
from app.database.db import get_db
from app.models.schemas import FaceResult, RecognizeResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recognize", tags=["Recognition"])

# ─── Shared singletons (loaded once) ─────────────────────────────────────────
_detector   = FaceDetector()
_embedder   = FaceEmbedder()
_recognizer = FaceRecognizer()

# Try to load a pre-built index at startup
_index_ready = _recognizer.load()
if not _index_ready:
    logger.warning(
        "No FAISS index found. Call POST /enroll first to build one."
    )


def _reload_recognizer() -> bool:
    """Reload the FAISS index from disk (picks up any freshly-enrolled data)."""
    loaded = _recognizer.load()
    if loaded:
        logger.info("FAISS index reloaded: %d vectors.", _recognizer.index.ntotal)
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "",
    response_model=RecognizeResponse,
    summary="Recognise faces in an uploaded image",
    description=(
        "Upload a JPEG/PNG image. The system detects all faces, queries the FAISS "
        "index using cosine similarity, marks attendance for known people, and logs "
        "unknown detections."
    ),
)
async def recognize_image(
    file: UploadFile = File(..., description="JPEG or PNG image containing one or more faces"),
    db:   Session    = Depends(get_db),
) -> RecognizeResponse:
    """
    Identify all faces in the uploaded image.

    - Detected faces are matched against the enrolled FAISS index.
    - Known faces → attendance marked automatically.
    - Unknown faces → logged to DB and image saved to storage/unknown_faces/.
    """
    global _index_ready

    # ── Always reload from disk so newly-enrolled data is reflected ────────
    # This is cheap (small index) and eliminates the stale-singleton bug.
    _index_ready = _reload_recognizer()
    if not _index_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recognition index not ready. Please call POST /enroll first.",
        )

    # ── Decode uploaded image ─────────────────────────────────────────────
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Primary decode path: OpenCV (fast, handles most JPEG/PNG/BMP)
    np_arr = np.frombuffer(contents, np.uint8)
    image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Fallback decode path: Pillow (handles WEBP, progressive JPEG, HEIC, etc.)
    if image is None:
        try:
            pil_img = PILImage.open(io.BytesIO(contents)).convert("RGB")
            image   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            logger.info("cv2.imdecode failed; decoded via Pillow fallback (format: %s).", pil_img.format)
        except Exception as pil_exc:
            logger.warning("Both cv2 and Pillow failed to decode image: %s", pil_exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Could not decode image. "
                    "Ensure the file is a valid JPEG, PNG, or WEBP image."
                ),
            )

    # ── Detect faces ──────────────────────────────────────────────────────
    faces, boxes = _detector.detect(image)

    if faces is None or boxes is None:
        return RecognizeResponse(
            success=True,
            faces_detected=0,
            results=[],
            message="No faces detected in the uploaded image.",
        )

    # ── Generate embeddings & recognise ───────────────────────────────────
    embeddings    = _embedder.get_embedding(faces)          # (N, 512)
    match_results = _recognizer.recognize_batch(embeddings) # [(name, conf), ...]

    attendance_mgr = AttendanceManager(db)
    face_results:  List[FaceResult] = []

    for i, (name, confidence) in enumerate(match_results):
        bbox = boxes[i].tolist() if i < len(boxes) else None
        is_unknown = name == UNKNOWN_LABEL

        if is_unknown:
            # Save unknown face crop for security logging
            if bbox:
                x1, y1, x2, y2 = bbox
                crop = image[y1:y2, x1:x2]
                attendance_mgr.log_unknown_face(face_crop=crop if crop.size > 0 else None)
            else:
                attendance_mgr.log_unknown_face()
        else:
            # Mark attendance (deduplication handled inside)
            attendance_mgr.mark_attendance(name, confidence)

        face_results.append(
            FaceResult(
                name=name,
                confidence=round(confidence, 4),
                is_unknown=is_unknown,
                bbox=bbox,
            )
        )

    return RecognizeResponse(
        success=True,
        faces_detected=len(face_results),
        results=face_results,
        message=f"Processed {len(face_results)} face(s) successfully.",
    )
