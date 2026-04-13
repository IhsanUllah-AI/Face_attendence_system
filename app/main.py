"""
main.py
-------
FastAPI application entry point.

Registers all route modules, configures CORS, initialises the database,
and provides a health-check endpoint.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from app.api import attendance, enroll, recognize
from app.api import dashboard
from app.config import API_TITLE, API_VERSION, BASE_DIR
from app.database.db import init_db
from app.models.schemas import HealthResponse

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "An end-to-end Facial Recognition Attendance System "
        "using MTCNN + FaceNet + FAISS + FastAPI."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Startup Event
# ─────────────────────────────────────────────
@app.on_event("startup")
async def on_startup() -> None:
    """Initialise DB tables on application startup."""
    init_db()
    logger.info("Database initialised.")
    logger.info("%s v%s is ready.", API_TITLE, API_VERSION)


# ─────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────
app.include_router(enroll.router)
app.include_router(recognize.router)
app.include_router(attendance.router)
app.include_router(dashboard.router)

# ─────────────────────────────────────────────
# Static Files (frontend assets)
# ─────────────────────────────────────────────
_frontend_dir = BASE_DIR / "frontend" / "assets"
_frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(_frontend_dir)), name="assets")


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
async def health_check() -> HealthResponse:
    """Return system health status."""
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        message="Facial Recognition Attendance System is running.",
    )


@app.get("/", tags=["System"], summary="Root redirect info")
async def root():
    return {
        "message": "Welcome to the Facial Recognition Attendance System API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }
