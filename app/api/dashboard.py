"""
dashboard.py
-------------
Serves the static frontend dashboard HTML and provides
extra utility endpoints:
  GET  /dashboard          → HTML page
  GET  /attendance/export  → Download attendance as CSV
  DELETE /attendance/{id}  → Delete a single record
  GET  /attendance/stats   → Summary statistics
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.database.db import AttendanceRecord, get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Dashboard"])

# ─────────────────────────────────────────────
# Serve HTML Dashboard
# ─────────────────────────────────────────────
_HTML_PATH = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Facial Recognition Attendance Dashboard",
    include_in_schema=False,
)
async def dashboard():
    """Serve the frontend web dashboard."""
    if not _HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return HTMLResponse(content=_HTML_PATH.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# Extra API: CSV Export
# ─────────────────────────────────────────────
@router.get(
    "/attendance/export",
    summary="Export attendance records to CSV",
    tags=["Attendance"],
)
def export_attendance_csv(
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
    name: Optional[str] = Query(None, description="Filter by name substring"),
    db: Session = Depends(get_db),
):
    """Download all attendance records as a CSV file."""
    query = db.query(AttendanceRecord)
    if date:
        query = query.filter(AttendanceRecord.date == date)
    if name:
        query = query.filter(AttendanceRecord.name.ilike(f"%{name}%"))

    records = query.order_by(AttendanceRecord.timestamp.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Name", "Date", "Time", "Confidence", "Timestamp"])
    for r in records:
        writer.writerow([r.id, r.name, r.date, r.time, r.confidence, r.timestamp])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance.csv"},
    )


# ─────────────────────────────────────────────
# Extra API: Delete Record
# ─────────────────────────────────────────────
@router.delete(
    "/attendance/{record_id}",
    summary="Delete an attendance record",
    tags=["Attendance"],
)
def delete_attendance_record(
    record_id: int,
    db: Session = Depends(get_db),
):
    """Delete a single attendance record by ID."""
    record = db.query(AttendanceRecord).filter(AttendanceRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record {record_id} not found.",
        )
    db.delete(record)
    db.commit()
    logger.info("Attendance record %d deleted.", record_id)
    return {"success": True, "message": f"Record {record_id} deleted."}


# ─────────────────────────────────────────────
# Extra API: Statistics
# ─────────────────────────────────────────────
@router.get(
    "/attendance/stats",
    summary="Attendance summary statistics",
    tags=["Attendance"],
)
def attendance_stats(db: Session = Depends(get_db)):
    """Return summary statistics for attendance dashboard cards."""
    from sqlalchemy import func
    import datetime

    today = datetime.date.today().strftime("%Y-%m-%d")

    total_records = db.query(func.count(AttendanceRecord.id)).scalar() or 0
    today_records = (
        db.query(func.count(AttendanceRecord.id))
        .filter(AttendanceRecord.date == today)
        .scalar()
        or 0
    )
    unique_persons = db.query(func.count(func.distinct(AttendanceRecord.name))).scalar() or 0

    avg_conf = db.query(func.avg(AttendanceRecord.confidence)).scalar()
    avg_conf = round(float(avg_conf), 4) if avg_conf else 0.0

    # Most frequent attendees (top 5)
    top_attendees = (
        db.query(AttendanceRecord.name, func.count(AttendanceRecord.id).label("count"))
        .group_by(AttendanceRecord.name)
        .order_by(func.count(AttendanceRecord.id).desc())
        .limit(5)
        .all()
    )

    return {
        "total_records": total_records,
        "today_records": today_records,
        "unique_persons": unique_persons,
        "average_confidence": avg_conf,
        "top_attendees": [{"name": row.name, "count": row.count} for row in top_attendees],
        "today": today,
    }
