"""
config.py
---------
Central configuration for the Facial Recognition Attendance System.
All tunable parameters are defined here to avoid hard-coded values.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Base Paths
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
STORAGE_DIR = BASE_DIR / "storage"

EMBEDDINGS_DIR    = STORAGE_DIR / "embeddings"
UNKNOWN_FACES_DIR = STORAGE_DIR / "unknown_faces"
LOGS_DIR          = STORAGE_DIR / "logs"
DB_PATH           = STORAGE_DIR / "attendance.db"

# ─────────────────────────────────────────────
# FAISS / Embedding Files
# ─────────────────────────────────────────────
FAISS_INDEX_PATH    = EMBEDDINGS_DIR / "face_index.faiss"
METADATA_JSON_PATH  = EMBEDDINGS_DIR / "metadata.json"

# ─────────────────────────────────────────────
# Model Settings
# ─────────────────────────────────────────────
EMBEDDING_DIM       = 512          # FaceNet InceptionResnetV1 output dimension
IMAGE_SIZE          = 160          # Required input size for FaceNet
MIN_FACE_SIZE       = 40           # Minimum face size (px) for MTCNN
MTCNN_THRESHOLDS    = [0.6, 0.7, 0.7]  # P, R, O-Net thresholds
MTCNN_MIN_MARGIN    = 14

# ─────────────────────────────────────────────
# Recognition Thresholds
# ─────────────────────────────────────────────
COSINE_SIMILARITY_THRESHOLD = 0.45  # Above this = recognised (centroid approach is more robust)
UNKNOWN_LABEL               = "Unknown"

# ─────────────────────────────────────────────
# Real-Time Processing
# ─────────────────────────────────────────────
WEBCAM_INDEX        = 0
FRAME_SKIP          = 2            # Process every Nth frame (performance)
DISPLAY_FPS         = True

# ─────────────────────────────────────────────
# Attendance
# ─────────────────────────────────────────────
ATTENDANCE_COOLDOWN_SECONDS = 60   # Min seconds between marking same person

# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────
API_HOST   = "0.0.0.0"
API_PORT   = 8000
API_TITLE  = "Facial Recognition Attendance System"
API_VERSION = "1.0.0"

# ─────────────────────────────────────────────
# Ensure all directories exist
# ─────────────────────────────────────────────
for _dir in [EMBEDDINGS_DIR, UNKNOWN_FACES_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
