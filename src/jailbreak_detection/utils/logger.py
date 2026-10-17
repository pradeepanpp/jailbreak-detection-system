# src/jailbreak_detection/utils/logger.py
"""Project logger: INFO to the console, DEBUG to logs/ (set LOG_TO_FILE=0 to turn off)."""

import os
import sys
from datetime import datetime

from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
)

if os.environ.get("LOG_TO_FILE", "1") != "0":
    try:
        os.makedirs("logs", exist_ok=True)
        logger.add(
            f"logs/jailbreak_{datetime.now():%Y%m%d}.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
            rotation="10 MB",
            retention="7 days",
        )
    except OSError as e:   # e.g. read-only file system in a container
        logger.warning(f"File logging disabled: {e}")
