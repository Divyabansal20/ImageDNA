"""
Central place for all project paths and settings.
If something changes later, we only update this file.
"""

from pathlib import Path

# Project root (ImageDNA/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset folders
DATASET_DIR = PROJECT_ROOT / "dataset"
REAL_DIR = DATASET_DIR / "real"
SCREEN_DIR = DATASET_DIR / "screen"

# Output folders
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"