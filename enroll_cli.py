"""
enroll_cli.py
--------------
CLI helper to run enrollment without starting the FastAPI server.

Usage:
    python enroll_cli.py

This will:
    1. Walk through Dataset/<person>/ folders
    2. Detect faces with MTCNN
    3. Generate FaceNet embeddings
    4. Build and save FAISS index to storage/embeddings/
"""

import sys
import logging
from pathlib import Path

# Fix Unicode output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.core.enrollment import EnrollmentManager

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Facial Recognition Attendance System - Enrollment CLI")
    print("=" * 60 + "\n")

    manager = EnrollmentManager()
    result  = manager.enroll_all()

    print("\n" + "-" * 60)
    print(f"  [OK] Enrollment complete!")
    print(f"  Persons   : {result['total_persons']}")
    print(f"  Embeddings: {result['total_embeddings']}")
    print(f"  Names     : {', '.join(result['persons'])}")
    print("-" * 60 + "\n")
