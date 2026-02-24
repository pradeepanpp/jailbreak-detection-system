# src/jailbreak_detection/utils/helpers.py
"""
Shared helper functions used across all modules.
"""

import yaml
import os
from pathlib import Path
from loguru import logger


def load_config(config_path: str = "configs/detection_config.yaml") -> dict:
    """
    Load YAML config file.
    Used by every module to read settings.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Config loaded from {config_path}")
    return config


def ensure_dirs(*paths: str):
    """
    Create directories if they don't exist.
    Call this at the start of any script that writes files.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root.
    Useful for building file paths that work from any directory.
    """
    return Path(__file__).parent.parent.parent.parent