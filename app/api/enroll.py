"""
enroll.py
---------
POST /enroll
    Triggers dataset processing pipeline:
    Detect faces → Generate FaceNet embeddings → Build & save FAISS index.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.enrollment import EnrollmentManager
from app.models.schemas import EnrollResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enroll", tags=["Enrollment"])


@router.post(
    "",
    response_model=EnrollResponse,
    summary="Build face embeddings index from dataset",
    description=(
        "Walks through the Dataset/ directory, detects faces with MTCNN, "
        "generates 512-d embeddings with FaceNet, and builds / saves a FAISS index."
    ),
)
async def enroll_dataset() -> EnrollResponse:
    """
    Process the entire dataset and build the FAISS recognition index.

    Returns
    -------
    EnrollResponse with enrollment statistics.
    """
    logger.info("Enrollment request received.")
    try:
        manager = EnrollmentManager()
        result  = manager.enroll_all()

        return EnrollResponse(
            success=True,
            total_persons=result["total_persons"],
            total_images=result["total_images"],
            persons=result["persons"],
            message=(
                f"Successfully enrolled {result['total_persons']} person(s) "
                f"with {result['total_images']} face embedding(s)."
            ),
        )

    except FileNotFoundError as exc:
        logger.error("Dataset not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during enrollment.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment failed: {exc}",
        )
