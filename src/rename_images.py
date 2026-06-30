"""
rename_images.py

Renames all images in the dataset to a clean and consistent format.
"""

from pathlib import Path

from src.config import REAL_DIR, SCREEN_DIR


def rename_images(folder: Path, prefix: str) -> None:
    """
    Renames all images inside a folder.

    Example:
    real_001.jpeg
    real_002.jpeg
    """

    images = sorted(folder.glob("*"))

    for index, image_path in enumerate(images, start=1):

        new_name = f"{prefix}_{index:03d}.jpeg"

        new_path = folder / new_name

        image_path.rename(new_path)

    print(f"✓ Renamed {len(images)} images inside '{folder.name}'")


def main():

    rename_images(REAL_DIR, "real")

    rename_images(SCREEN_DIR, "screen")

    print("\n✓ Dataset renamed successfully.")


if __name__ == "__main__":
    main()