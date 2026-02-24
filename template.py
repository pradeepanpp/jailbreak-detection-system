# template.py
"""
Project structure generator for Jailbreak Detection System.
Run once after cloning the repo.

Usage: python template.py
"""

import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s'
)

# -----------------------------------------------
# PROJECT NAME
# -----------------------------------------------
project_name = "jailbreak_detection"

# -----------------------------------------------
# ALL FILES AND FOLDERS
# -----------------------------------------------
list_of_files = [

    # GitHub Actions
    ".github/workflows/.gitkeep",

    # Source package
    "src/__init__.py",
    f"src/{project_name}/__init__.py",

    # Preprocessing
    f"src/{project_name}/preprocessing/__init__.py",
    f"src/{project_name}/preprocessing/normalizer.py",
    f"src/{project_name}/preprocessing/decoder.py",

    # Detection layers
    f"src/{project_name}/detection/__init__.py",
    f"src/{project_name}/detection/rule_engine.py",
    f"src/{project_name}/detection/ml_classifier.py",
    f"src/{project_name}/detection/embedding_matcher.py",
    f"src/{project_name}/detection/llm_judge.py",

    # Aggregator
    f"src/{project_name}/aggregator/__init__.py",
    f"src/{project_name}/aggregator/risk_engine.py",

    # API
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/main.py",
    f"src/{project_name}/api/schemas.py",
    f"src/{project_name}/api/routes.py",

    # Dashboard
    f"src/{project_name}/dashboard/__init__.py",
    f"src/{project_name}/dashboard/app.py",
    f"src/{project_name}/dashboard/components.py",

    # Utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/logger.py",
    f"src/{project_name}/utils/helpers.py",

    # Constants
    f"src/{project_name}/constants/__init__.py",

    # Scripts
    "scripts/download_datasets.py",
    "scripts/train_classifier.py",
    "scripts/build_index.py",
    "scripts/run_evaluation.py",

    # Tests
    "tests/__init__.py",
    "tests/test_preprocessing.py",
    "tests/test_rule_engine.py",
    "tests/test_classifier.py",
    "tests/test_embedding_matcher.py",
    "tests/test_aggregator.py",
    "tests/test_api.py",

    # Configs
    "configs/detection_config.yaml",

    # Data directories
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/results/.gitkeep",

    # Models
    "models/.gitkeep",

    # Docker
    "Dockerfile",
    "docker-compose.yml",

    # Root files
    "requirements.txt",
    "setup.py",
    ".env.example",
    "README.md",
]

# -----------------------------------------------
# CREATE FILES AND FOLDERS
# -----------------------------------------------
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory : {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created empty file : {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")

logging.info("\n✅ Project structure created successfully!")
logging.info("\n📦 Setup with UV (Recommended):")
logging.info("   uv venv")
logging.info("   source venv/bin/activate  # Mac/Linux")
logging.info("   venv\\Scripts\\activate     # Windows")
logging.info("   uv pip install -r requirements.txt")