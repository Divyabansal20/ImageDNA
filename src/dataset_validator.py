"""
dataset_validator.py

Checks whether the dataset is valid before training.
No images are modified here.
"""

from pathlib import Path
from typing import Dict

from PIL import Image

from src.config import REAL_DIR, SCREEN_DIR


def validate_folder(folder: Path) -> list[Path]:
    """
    Returns all image files inside a folder.

    Raises:
        FileNotFoundError:
            If the folder does not exist.
    """

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_files = sorted(folder.glob("*"))

    return image_files


def inspect_images(image_paths: list[Path]) -> Dict:
    """
    Reads every image and collects basic statistics.
    """

    corrupted = 0
    rgb_images = 0

    widths = []
    heights = []

    formats = {}

    for image_path in image_paths:

        try:
            with Image.open(image_path) as img:

                widths.append(img.width)
                heights.append(img.height)

                if img.mode == "RGB":
                    rgb_images += 1

                image_format = img.format

                formats[image_format] = formats.get(image_format, 0) + 1

        except Exception:
            corrupted += 1

    return {

        "total": len(image_paths),

        "corrupted": corrupted,

        "rgb": rgb_images,

        "formats": formats,

        "min_width": min(widths),

        "max_width": max(widths),

        "min_height": min(heights),

        "max_height": max(heights),

        "avg_width": sum(widths) / len(widths),

        "avg_height": sum(heights) / len(heights),

    }


def print_report(real_stats: Dict, screen_stats: Dict):
    """
    Prints a clean validation report.
    """

    print("\n" + "=" * 55)
    print("           ImageDNA Dataset Report")
    print("=" * 55)

    print(f"\nReal Images   : {real_stats['total']}")
    print(f"Screen Images : {screen_stats['total']}")

    print(f"\nCorrupted (Real)   : {real_stats['corrupted']}")
    print(f"Corrupted (Screen) : {screen_stats['corrupted']}")

    print("\nFormats")

    print(f"Real   : {real_stats['formats']}")
    print(f"Screen : {screen_stats['formats']}")

    print("\nResolution")

    print(
        f"Real   : {real_stats['min_width']}x{real_stats['min_height']} "
        f"to {real_stats['max_width']}x{real_stats['max_height']}"
    )

    print(
        f"Screen : {screen_stats['min_width']}x{screen_stats['min_height']} "
        f"to {screen_stats['max_width']}x{screen_stats['max_height']}"
    )

    print("\nAverage Resolution")

    print(
        f"Real   : {real_stats['avg_width']:.1f} x {real_stats['avg_height']:.1f}"
    )

    print(
        f"Screen : {screen_stats['avg_width']:.1f} x {screen_stats['avg_height']:.1f}"
    )

    print("\nRGB Images")

    print(f"Real   : {real_stats['rgb']}")
    print(f"Screen : {screen_stats['rgb']}")

    print("\nDataset validation completed successfully.")
    print("=" * 55)


def validate_dataset():
    """
    Runs the complete validation pipeline.
    """

    real_images = validate_folder(REAL_DIR)
    screen_images = validate_folder(SCREEN_DIR)

    real_stats = inspect_images(real_images)
    screen_stats = inspect_images(screen_images)

    print_report(real_stats, screen_stats)


if __name__ == "__main__":
    validate_dataset()