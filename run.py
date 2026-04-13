"""
run.py
------
Application entry point.
Launches the FastAPI server using Uvicorn with environment-based configuration.

Usage:
    python run.py

Environment Variables:
    API_HOST         : Server host (default: 0.0.0.0)
    API_PORT         : Server port (default: 8000)
    ENVIRONMENT      : Environment mode (development/production, default: development)
    DEBUG            : Enable debug mode (default: false)
    LOG_LEVEL        : Logging level (default: INFO)
"""

import logging
import uvicorn
from app.config import API_HOST, API_PORT, ENVIRONMENT, DEBUG, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Facial Recognition Attendance System...")
    logger.info("Host: %s, Port: %d, Environment: %s", API_HOST, API_PORT, ENVIRONMENT)
    
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,  # Auto-reload only in development mode
        log_level=LOG_LEVEL.lower(),
    )
