# src/jailbreak_detection/utils/helpers.py
"""
Shared helper functions used across all modules.
"""

import yaml
import os
from pathlib import Path
from loguru import logger


DEFAULT_CONFIG = "configs/detection_config.yaml"


def load_config(config_path: str = DEFAULT_CONFIG) -> dict:
    """
    Load YAML config file.

    Relative paths are tried from the current directory first, then from
    the project root, so scripts and tests work from any directory.
    """
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = get_project_root() / config_path
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Config loaded from {path}")
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