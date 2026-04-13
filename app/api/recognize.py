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

from app.config import UNKNOWN_LABEL, MAX_UPLOAD_SIZE_BYTES, REQUEST_TIMEOUT_SECONDS
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

# Load initial index (or log warning if not ready)
_index_loaded = _recognizer.load()
if not _index_loaded:
    logger.warning(
        "No FAISS index found at startup. Call POST /enroll first to build one."
    )


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
    
    Raises
    ------
    HTTPException
        400: Invalid/empty file or too large
        503: Recognition index not ready (call POST /enroll first)
    """
    # ── Validate file size before reading ──────────────────────────────────
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f}MB limit.",
        )

    # ── Reload recognizer from disk (picks up newly enrolled faces) ───────
    # This is a deliberate trade-off: reload cost vs. freshness guarantee.
    # For high-throughput scenarios, consider caching with TTL.
    loaded = _recognizer.load()
    if not loaded or _recognizer.index is None:
        logger.error("Failed to load FAISS index from disk.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recognition index not ready. Please call POST /enroll first.",
        )

    # ── Decode uploaded image (with fallback) ─────────────────────────────
    np_arr = np.frombuffer(contents, np.uint8)
    image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Fallback: use Pillow for formats cv2 can't handle (WEBP, progressive JPEG, HEIC)
    if image is None:
        try:
            pil_img = PILImage.open(io.BytesIO(contents)).convert("RGB")
            image   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            logger.debug(
                "cv2.imdecode fallback used (format: %s).",
                getattr(pil_img, "format", "unknown"),
            )
        except Exception as pil_exc:
            logger.warning("Image decode failed (cv2 and Pillow): %s", pil_exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Could not decode image. "
                    "Ensure the file is a valid JPEG, PNG, or WEBP image."
                ),
            )

    # ── Detect faces ──────────────────────────────────────────────────────
    try:
        faces, boxes = _detector.detect(image)
    except Exception as det_exc:
        logger.error("Face detection failed: %s", det_exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face detection failed. Please try again.",
        )

    if faces is None or boxes is None:
        return RecognizeResponse(
            success=True,
            faces_detected=0,
            results=[],
            message="No faces detected in the uploaded image.",
        )

    # ── Generate embeddings & recognise ───────────────────────────────────
    try:
        embeddings    = _embedder.get_embedding(faces)          # (N, 512)
        match_results = _recognizer.recognize_batch(embeddings) # [(name, conf), ...]
    except Exception as rec_exc:
        logger.error("Recognition/embedding failed: %s", rec_exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face recognition failed. Please try again.",
        )

    attendance_mgr = AttendanceManager(db)
    face_results:  List[FaceResult] = []

    for i, (name, confidence) in enumerate(match_results):
        bbox = boxes[i].tolist() if i < len(boxes) else None
        is_unknown = name == UNKNOWN_LABEL

        try:
            if is_unknown:
                # Save unknown face crop for security logging
                if bbox:
                    x1, y1, x2, y2 = bbox
                    crop = image[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        attendance_mgr.log_unknown_face(face_crop=crop)
                    else:
                        attendance_mgr.log_unknown_face()
                else:
                    attendance_mgr.log_unknown_face()
            else:
                # Mark attendance (deduplication handled inside)
                attendance_mgr.mark_attendance(name, confidence)
                logger.debug("Attendance marked for %s (confidence: %.4f)", name, confidence)
        except Exception as att_exc:
            logger.error("Attendance marking failed for %s: %s", name, att_exc)
            # Continue processing other faces even if one fails

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
