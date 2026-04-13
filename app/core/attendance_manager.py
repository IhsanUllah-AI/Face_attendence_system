"""
attendance_manager.py
----------------------
Attendance Management module.

Responsibilities:
    - Mark attendance for a recognised person (once per day / cooldown).
    - Log unknown-face detections and save their images.
    - Provide attendance query helpers.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.config import ATTENDANCE_COOLDOWN_SECONDS, UNKNOWN_FACES_DIR, UNKNOWN_LABEL
from app.database.db import AttendanceRecord, UnknownFaceLog

logger = logging.getLogger(__name__)


class AttendanceManager:
    """
    Handles attendance marking and unknown-face logging.

    Parameters
    ----------
    db : Session
        SQLAlchemy database session.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ─────────────────────────────────────────
    # Attendance
    # ─────────────────────────────────────────

    def mark_attendance(
        self,
        name: str,
        confidence: float,
    ) -> Optional[AttendanceRecord]:
        """
        Mark attendance for a recognised person.

        Rules:
            - One record per person per calendar day.
            - Cooldown: at least ATTENDANCE_COOLDOWN_SECONDS since last mark.

        Parameters
        ----------
        name       : str   Recognised person's name.
        confidence : float Cosine similarity score.

        Returns
        -------
        AttendanceRecord if a new entry was created, else None.
        """
        now  = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        # Check if already marked today
        existing = (
            self.db.query(AttendanceRecord)
            .filter(AttendanceRecord.name == name, AttendanceRecord.date == date)
            .first()
        )

        if existing:
            # Check cooldown
            last_ts = existing.timestamp
            elapsed = (now - last_ts).total_seconds()
            if elapsed < ATTENDANCE_COOLDOWN_SECONDS:
                logger.debug(
                    "Attendance for '%s' skipped (cooldown: %.0fs remaining).",
                    name,
                    ATTENDANCE_COOLDOWN_SECONDS - elapsed,
                )
                return None

        record = AttendanceRecord(
            name=name,
            date=date,
            time=time,
            confidence=round(confidence, 4),
            timestamp=now,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)

        logger.info("Attendance marked → %s [%s %s] (conf=%.3f)", name, date, time, confidence)
        return record

    def get_attendance(
        self,
        date_filter: Optional[str] = None,
        name_filter: Optional[str] = None,
        limit: int = 500,
    ) -> List[AttendanceRecord]:
        """
        Fetch attendance records with optional filters.

        Parameters
        ----------
        date_filter : 'YYYY-MM-DD' string or None
        name_filter : person name substring or None
        limit       : max records to return
        """
        query = self.db.query(AttendanceRecord)

        if date_filter:
            query = query.filter(AttendanceRecord.date == date_filter)
        if name_filter:
            query = query.filter(AttendanceRecord.name.ilike(f"%{name_filter}%"))

        return query.order_by(AttendanceRecord.timestamp.desc()).limit(limit).all()

    # ─────────────────────────────────────────
    # Unknown Face Handling
    # ─────────────────────────────────────────

    def log_unknown_face(
        self,
        face_crop: Optional[np.ndarray] = None,
    ) -> UnknownFaceLog:
        """
        Log an unknown face detection. Optionally saves face image to disk.

        Parameters
        ----------
        face_crop : BGR ndarray | None
            Cropped face image to save. If None, only a log entry is created.

        Returns
        -------
        UnknownFaceLog database record.
        """
        image_path: Optional[str] = None

        if face_crop is not None:
            filename = f"unknown_{uuid.uuid4().hex[:8]}_{datetime.datetime.now():%Y%m%d_%H%M%S}.jpg"
            save_path = UNKNOWN_FACES_DIR / filename
            cv2.imwrite(str(save_path), face_crop)
            image_path = str(save_path)
            logger.warning("Unknown face saved → %s", save_path)
        else:
            logger.warning("Unknown face detected (no image saved).")

        log_entry = UnknownFaceLog(image_path=image_path, alerted=False)
        self.db.add(log_entry)
        self.db.commit()
        self.db.refresh(log_entry)

        # ── Optional: trigger alert (extendable) ──
        self._trigger_alert(image_path)

        return log_entry

    # ─────────────────────────────────────────
    # Alerts (extendable hook)
    # ─────────────────────────────────────────

    @staticmethod
    def _trigger_alert(image_path: Optional[str]) -> None:
        """
        Alert hook for unknown face detections.
        Currently logs to console. Extend to send email / push notification.
        """
        msg = f"[SECURITY ALERT] Unknown face detected!"
        if image_path:
            msg += f" Image: {image_path}"
        logger.warning(msg)
        print(msg)
