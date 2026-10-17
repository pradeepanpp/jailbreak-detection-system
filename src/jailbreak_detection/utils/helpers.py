# src/jailbreak_detection/utils/helpers.py
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = "configs/detection_config.yaml"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(config_path: str = DEFAULT_CONFIG) -> dict:
    """Read the YAML config. Relative paths also work when run from another directory."""
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = get_project_root() / config_path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: str):
    for path in paths:
        os.makedirs(path, exist_ok=True)
