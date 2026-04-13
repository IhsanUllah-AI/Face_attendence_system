"""
attendance.py
-------------
GET /attendance
    Return stored attendance records with optional filters.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.attendance_manager import AttendanceManager
from app.database.db import get_db
from app.models.schemas import AttendanceEntry, AttendanceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.get(
    "",
    response_model=AttendanceResponse,
    summary="Fetch attendance logs",
    description=(
        "Returns attendance records. Optional query params: "
        "`date` (YYYY-MM-DD), `name` (substring match), `limit`."
    ),
)
def get_attendance(
    date:  Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    name:  Optional[str] = Query(None, description="Filter by person name (substring)"),
    limit: int           = Query(200, ge=1, le=1000, description="Max records to return"),
    db:    Session       = Depends(get_db),
) -> AttendanceResponse:
    """
    Retrieve attendance logs.

    Query Parameters
    ----------------
    date  : Filter records for a specific date (YYYY-MM-DD).
    name  : Filter by person name (case-insensitive substring match).
    limit : Maximum number of records (default 200, max 1000).
    """
    mgr     = AttendanceManager(db)
    records = mgr.get_attendance(date_filter=date, name_filter=name, limit=limit)

    entries = [
        AttendanceEntry(
            id=r.id,
            name=r.name,
            date=r.date,
            time=r.time,
            confidence=r.confidence,
            timestamp=r.timestamp,
        )
        for r in records
    ]

    return AttendanceResponse(
        success=True,
        total=len(entries),
        records=entries,
    )
