# src/jailbreak_detection/utils/logger.py
"""
Centralized logger for the entire project.
Import this instead of using print() anywhere.
"""

import os
import sys
from loguru import logger
from datetime import datetime

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Remove default logger
logger.remove()

# Console logger — colored output
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan> - "
           "<level>{message}</level>",
    level="INFO"
)

# File logger — full details saved to disk
logger.add(
    f"logs/jailbreak_{datetime.now().strftime('%Y%m%d')}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days"
)