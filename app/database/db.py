"""
db.py
-----
Database setup using SQLAlchemy with SQLite.
Manages the Attendance table and Unknown-face detection log.
"""

import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.config import DB_PATH

# ─────────────────────────────────────────────
# Engine & Session
# ─────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
class AttendanceRecord(Base):
    """Stores one attendance entry per person per day."""
    __tablename__ = "attendance"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String, nullable=False, index=True)
    date        = Column(String, nullable=False)   # YYYY-MM-DD
    time        = Column(String, nullable=False)   # HH:MM:SS
    confidence  = Column(Float, nullable=False)
    timestamp   = Column(DateTime, default=datetime.datetime.utcnow)


class UnknownFaceLog(Base):
    """Stores log entries for unrecognised faces."""
    __tablename__ = "unknown_faces"

    id           = Column(Integer, primary_key=True, index=True)
    image_path   = Column(String, nullable=True)   # Saved image path
    detected_at  = Column(DateTime, default=datetime.datetime.utcnow)
    alerted      = Column(Boolean, default=False)


# ─────────────────────────────────────────────
# Initialise tables
# ─────────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


# ─────────────────────────────────────────────
# Dependency helper (used in FastAPI routes)
# ─────────────────────────────────────────────
def get_db():
    """Yield a DB session and ensure it is closed after use."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
