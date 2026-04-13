"""
schemas.py
----------
Pydantic request/response schemas for FastAPI endpoints.
"""

from __future__ import annotations

import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enroll
# ─────────────────────────────────────────────
class EnrollResponse(BaseModel):
    """Response returned after building the FAISS index."""
    success       : bool
    total_persons : int
    total_images  : int
    persons       : List[str]
    message       : str


# ─────────────────────────────────────────────
# Recognize
# ─────────────────────────────────────────────
class FaceResult(BaseModel):
    """Result for a single detected face in a frame."""
    name        : str
    confidence  : float = Field(..., ge=0.0, le=1.0)
    is_unknown  : bool
    bbox        : Optional[List[int]] = None   # [x1, y1, x2, y2]


class RecognizeResponse(BaseModel):
    """Response returned by the /recognize endpoint."""
    success       : bool
    faces_detected: int
    results       : List[FaceResult]
    message       : str


# ─────────────────────────────────────────────
# Attendance
# ─────────────────────────────────────────────
class AttendanceEntry(BaseModel):
    """Single attendance record."""
    id          : int
    name        : str
    date        : str
    time        : str
    confidence  : float
    timestamp   : datetime.datetime

    class Config:
        from_attributes = True


class AttendanceResponse(BaseModel):
    """Response returned by GET /attendance."""
    success : bool
    total   : int
    records : List[AttendanceEntry]


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status  : str
    version : str
    message : str
