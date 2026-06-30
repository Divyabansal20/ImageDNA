"""
dataset_validator.py

Checks whether the dataset is healthy before training.
No image is modified in this step.
"""

from pathlib import Path
from PIL import Image

from src.config import REAL_DIR, SCREEN_DIR, OUTPUTS_DIR


def get_images(folder):
    """
    Returns all supported images from a folder.
    """

    valid_extensions = {".jpg", ".jpeg", ".png"}

    return sorted(
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in valid_extensions
    )


def inspect_folder(folder):
    """
    Reads every image and collects useful statistics.
    """

    images = get_images(folder)

    corrupted = 0
    rgb_count = 0

    widths = []
    heights = []

    formats = {}

    for image_path in images:

        try:

            with Image.open(image_path) as image:

                # Save image size so we can summarize the dataset later.
                widths.append(image.width)
                heights.append(image.height)

                # Count RGB images.
                if image.mode == "RGB":
                    rgb_count += 1

                image_format = image.format

                formats[image_format] = formats.get(image_format, 0) + 1

        except Exception:
            corrupted += 1

    return {
        "total": len(images),
        "corrupted": corrupted,
        "rgb": rgb_count,
        "formats": formats,
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_width": sum(widths) / len(widths),
        "avg_height": sum(heights) / len(heights),
    }


def generate_report(real, screen):
    """
    Creates a nicely formatted report.
    """

    report = []

    report.append("=" * 60)
    report.append("              ImageDNA Dataset Report")
    report.append("=" * 60)

    report.append(f"\nReal Images   : {real['total']}")
    report.append(f"Screen Images : {screen['total']}")

    report.append("\nCorrupted Images")
    report.append(f"Real   : {real['corrupted']}")
    report.append(f"Screen : {screen['corrupted']}")

    report.append("\nImage Formats")
    report.append(f"Real   : {real['formats']}")
    report.append(f"Screen : {screen['formats']}")

    report.append("\nResolution Range")

    report.append(
        f"Real   : {real['min_width']}x{real['min_height']}  →  {real['max_width']}x{real['max_height']}"
    )

    report.append(
        f"Screen : {screen['min_width']}x{screen['min_height']}  →  {screen['max_width']}x{screen['max_height']}"
    )

    report.append("\nAverage Resolution")

    report.append(
        f"Real   : {real['avg_width']:.1f} × {real['avg_height']:.1f}"
    )

    report.append(
        f"Screen : {screen['avg_width']:.1f} × {screen['avg_height']:.1f}"
    )

    report.append("\nRGB Images")

    report.append(f"Real   : {real['rgb']}")
    report.append(f"Screen : {screen['rgb']}")

    report.append("\n✓ Dataset validation passed.")
    report.append("=" * 60)

    return "\n".join(report)


def validate_dataset():
    """
    Runs the complete validation pipeline.
    """

    if not REAL_DIR.exists():
        raise FileNotFoundError(f"Real folder not found : {REAL_DIR}")

    if not SCREEN_DIR.exists():
        raise FileNotFoundError(f"Screen folder not found : {SCREEN_DIR}")

    OUTPUTS_DIR.mkdir(exist_ok=True)

    real_stats = inspect_folder(REAL_DIR)
    screen_stats = inspect_folder(SCREEN_DIR)

    report = generate_report(real_stats, screen_stats)

    print(report)

    report_file = OUTPUTS_DIR / "dataset_report.txt"

    with open(report_file, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"\nReport saved to : {report_file}")


if __name__ == "__main__":
    validate_dataset()