"""
config.py
---------
Central configuration for the Facial Recognition Attendance System.
All tunable parameters are defined here to avoid hard-coded values.
Loads from environment variables with sensible defaults.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load environment variables from .env file
# ─────────────────────────────────────────────
load_dotenv()

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
EMBEDDING_DIM       = int(os.getenv("EMBEDDING_DIM", "512"))
IMAGE_SIZE          = int(os.getenv("IMAGE_SIZE", "160"))
MIN_FACE_SIZE       = int(os.getenv("MIN_FACE_SIZE", "40"))
MTCNN_MIN_CONFIDENCE = float(os.getenv("MTCNN_MIN_CONFIDENCE", "0.80"))
MTCNN_THRESHOLDS    = [0.5, 0.6, 0.6]  # relaxed for better recall
MTCNN_MIN_MARGIN    = 14

# ─────────────────────────────────────────────
# Recognition Thresholds
# ─────────────────────────────────────────────
COSINE_SIMILARITY_THRESHOLD = float(os.getenv("COSINE_SIMILARITY_THRESHOLD", "0.45"))
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
ATTENDANCE_COOLDOWN_SECONDS = int(os.getenv("ATTENDANCE_COOLDOWN_SECONDS", "60"))

# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────
API_HOST   = os.getenv("API_HOST", "0.0.0.0")
API_PORT   = int(os.getenv("API_PORT", "8000"))
API_TITLE  = os.getenv("API_TITLE", "Facial Recognition Attendance System")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG      = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

# ─────────────────────────────────────────────
# CORS Configuration
# ─────────────────────────────────────────────
_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
ALLOWED_ORIGINS = [o.strip() for o in _origins_str.split(",")]
ALLOWED_METHODS = os.getenv("ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
ALLOWED_HEADERS = os.getenv("ALLOWED_HEADERS", "Content-Type,Authorization").split(",")

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# ─────────────────────────────────────────────
# File Upload Limits
# ─────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─────────────────────────────────────────────
# Ensure all directories exist
# ─────────────────────────────────────────────
for _dir in [EMBEDDINGS_DIR, UNKNOWN_FACES_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
