"""
db.py
-----
Database setup using SQLAlchemy with SQLite.
Manages the Attendance table and Unknown-face detection log.
Supports both SQLite and other SQL databases via environment variables.
"""

import datetime
import logging
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.config import DATABASE_URL, DB_PATH

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Engine & Session
# ─────────────────────────────────────────────
# SQLite-specific optimizations for SQLite
if "sqlite" in DATABASE_URL.lower():
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI
        poolclass=StaticPool,
        echo=False,  # Set to True for SQL query logging in debug
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Validate connections before using
        echo=False,
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
    date        = Column(String, nullable=False, index=True)   # YYYY-MM-DD
    time        = Column(String, nullable=False)   # HH:MM:SS
    confidence  = Column(Float, nullable=False)
    timestamp   = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    def __repr__(self) -> str:
        return (
            f"<AttendanceRecord(id={self.id}, name='{self.name}', "
            f"date='{self.date}', time='{self.time}', confidence={self.confidence})>"
        )


class UnknownFaceLog(Base):
    """Stores log entries for unrecognised faces."""
    __tablename__ = "unknown_faces"

    id           = Column(Integer, primary_key=True, index=True)
    image_path   = Column(String, nullable=True)   # Saved image path
    detected_at  = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    alerted      = Column(Boolean, default=False)

    def __repr__(self) -> str:
        return (
            f"<UnknownFaceLog(id={self.id}, image_path='{self.image_path}', "
            f"detected_at='{self.detected_at}', alerted={self.alerted})>"
        )


# ─────────────────────────────────────────────
# Initialise tables
# ─────────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they don't already exist."""
    try:
        # Ensure database directory exists (for SQLite)
        if "sqlite" in DATABASE_URL.lower():
            db_path = Path(DB_PATH)
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        Base.metadata.create_all(bind=engine)
        logger.info(
            "Database tables initialised successfully (URL: %s).",
            DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
        )
    except Exception as exc:
        logger.error("Failed to initialise database: %s", exc)
        raise


# ─────────────────────────────────────────────
# Dependency helper (used in FastAPI routes)
# ─────────────────────────────────────────────
def get_db() -> Session:
    """
    Yield a DB session and ensure it is closed after use.
    
    This is a FastAPI dependency that provides a database session
    to route handlers. The session is automatically closed after
    the request completes.
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception as exc:
        logger.error("Database session error: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()
