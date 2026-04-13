"""
realtime.py
------------
Real-Time Face Recognition via webcam.

Run this script directly (separate from the FastAPI server):
    python realtime.py

Controls:
    Q  → Quit
    R  → Force-reload the FAISS index (useful after enrolling new faces)

Pipeline per frame:
    1. Capture frame from webcam.
    2. Every FRAME_SKIP frames → run MTCNN detection.
    3. Generate FaceNet embeddings.
    4. Query FAISS index for cosine-similarity match.
    5. Draw bounding boxes + labels on frame.
    6. Mark attendance / log unknown faces in DB.
"""

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project root is on path when running as a script
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import (
    COSINE_SIMILARITY_THRESHOLD,
    DISPLAY_FPS,
    FRAME_SKIP,
    UNKNOWN_LABEL,
    WEBCAM_INDEX,
)
from app.core.attendance_manager import AttendanceManager
from app.core.detector import FaceDetector
from app.core.embedder import FaceEmbedder
from app.core.recognizer import FaceRecognizer
from app.database.db import SessionLocal, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("realtime")

# ─────────────────────────────────────────────
# Colour palette for bounding boxes
# ─────────────────────────────────────────────
COLOUR_KNOWN   = (0, 220, 0)    # Green  – known person
COLOUR_UNKNOWN = (0, 0, 220)    # Red    – unknown
COLOUR_FPS     = (200, 200, 0)  # Yellow – FPS overlay


def draw_annotations(
    frame: np.ndarray,
    name: str,
    confidence: float,
    box: np.ndarray,
) -> None:
    """Draw a labelled bounding box on the frame in-place."""
    x1, y1, x2, y2 = box.astype(int)
    colour  = COLOUR_KNOWN if name != UNKNOWN_LABEL else COLOUR_UNKNOWN
    label   = f"{name} ({confidence:.2f})"

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(
        frame, label,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1, cv2.LINE_AA,
    )


def run_realtime() -> None:
    """Main loop for real-time face recognition."""
    # ── Initialise models ────────────────────────────────────────────────
    logger.info("Loading models …")
    detector   = FaceDetector()
    embedder   = FaceEmbedder()
    recognizer = FaceRecognizer()

    if not recognizer.load():
        logger.error(
            "No FAISS index found! Run POST /enroll first (or python -c "
            "\"from app.core.enrollment import EnrollmentManager; "
            "EnrollmentManager().enroll_all()\")."
        )
        return

    # ── Initialise DB ────────────────────────────────────────────────────
    init_db()
    db = SessionLocal()
    attendance_mgr = AttendanceManager(db)

    # ── Open webcam ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        logger.error("Cannot open webcam (index %d).", WEBCAM_INDEX)
        db.close()
        return

    logger.info("Webcam opened. Press Q to quit, R to reload index.")

    frame_count    = 0
    last_results   = []   # Cache last detection results for non-detection frames
    last_boxes     = []

    # FPS tracking
    fps_timer      = time.time()
    fps_counter    = 0
    display_fps    = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam.")
                break

            frame_count += 1
            fps_counter  += 1

            # ── FPS calculation ──────────────────────────────────────────
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer   = time.time()

            # ── Detection every FRAME_SKIP frames ───────────────────────
            if frame_count % FRAME_SKIP == 0:
                faces, boxes = detector.detect(frame)

                if faces is not None and boxes is not None:
                    embeddings = embedder.get_embedding(faces)
                    results    = recognizer.recognize_batch(embeddings)

                    last_results = results
                    last_boxes   = boxes

                    # Attendance / unknown logging
                    for i, (name, conf) in enumerate(results):
                        crop = None
                        if i < len(boxes):
                            x1, y1, x2, y2 = boxes[i]
                            c = frame[y1:y2, x1:x2]
                            crop = c if c.size > 0 else None

                        if name == UNKNOWN_LABEL:
                            attendance_mgr.log_unknown_face(face_crop=crop)
                        else:
                            attendance_mgr.mark_attendance(name, conf)
                else:
                    last_results = []
                    last_boxes   = []

            # ── Draw annotations ────────────────────────────────────────
            for i, (name, conf) in enumerate(last_results):
                if i < len(last_boxes):
                    draw_annotations(frame, name, conf, last_boxes[i])

            # ── FPS overlay ─────────────────────────────────────────────
            if DISPLAY_FPS:
                cv2.putText(
                    frame, f"FPS: {display_fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COLOUR_FPS, 2, cv2.LINE_AA,
                )

            # ── Timestamp overlay ────────────────────────────────────────
            ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
            cv2.putText(
                frame, ts,
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA,
            )

            cv2.imshow("Facial Recognition Attendance System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("Quit signal received.")
                break
            elif key == ord("r"):
                logger.info("Reloading FAISS index …")
                recognizer.load()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()
        logger.info("Resources released. Goodbye.")


if __name__ == "__main__":
    run_realtime()
