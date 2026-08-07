from pathlib import Path
import sys

from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
)

logger.add(
    LOG_DIR / "graphagent.log",
    level="DEBUG",
    rotation="10 MB",      # Create a new log file after 10 MB
    retention="10 days",   # Keep logs for 10 days
    compression="zip",     # Compress old log files
    enqueue=True,          # Thread-safe logging
)
__all__ = ["logger"]