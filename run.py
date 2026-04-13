"""
run.py
------
Application entry point.
Launches the FastAPI server using Uvicorn.

Usage:
    python run.py
"""

import uvicorn
from app.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,         # Auto-reload on code changes (dev mode)
        log_level="info",
    )
