"""
config.py

Stores all project paths and a few common settings.
If a folder name changes in the future, update it here only.
"""

from pathlib import Path

# Root directory of the project (ImageDNA/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Paths

DATASET_DIR = PROJECT_ROOT / "dataset"

REAL_DIR = DATASET_DIR / "real"
SCREEN_DIR = DATASET_DIR / "screen"

# Output Paths

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Image Settings

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png"}

DEFAULT_IMAGE_SIZE = (512, 512)