"""
preprocessing.py

Loads an image and prepares different versions of it
for forensic feature extraction.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import DEFAULT_IMAGE_SIZE


@dataclass
class ProcessedImage:
    """
    Stores every version of an image that will be used
    by different forensic modules.
    """

    rgb: np.ndarray
    gray: np.ndarray
    normalized_rgb: np.ndarray
    normalized_gray: np.ndarray


def load_image(image_path: Path) -> np.ndarray:
    """
    Reads an image from disk and converts it to RGB.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # OpenCV loads images in BGR format.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def convert_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """
    Creates a grayscale version of the RGB image.
    """

    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Converts pixel values from 0-255 to 0-1.
    """

    return image.astype(np.float32) / 255.0


def preprocess_image(image_path: Path) -> ProcessedImage:
    """
    Complete preprocessing pipeline.

    Returns every version of the image that later
    feature extractors will need.
    """

    rgb_image = load_image(image_path)

    # Standardize image size to DEFAULT_IMAGE_SIZE using area interpolation
    rgb_image = cv2.resize(rgb_image, DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    gray_image = convert_to_grayscale(rgb_image)

    normalized_rgb = normalize_image(rgb_image)

    normalized_gray = normalize_image(gray_image)

    return ProcessedImage(
        rgb=rgb_image,
        gray=gray_image,
        normalized_rgb=normalized_rgb,
        normalized_gray=normalized_gray,
    )


def show_preprocessing(image_path: Path) -> None:
    """
    Displays every preprocessing stage.

    Useful for debugging and demonstrations.
    """

    processed = preprocess_image(image_path)

    plt.figure(figsize=(14, 4))

    # Original RGB image
    plt.subplot(1, 3, 1)
    plt.imshow(processed.rgb)
    plt.title("Original RGB")
    plt.axis("off")

    # Grayscale image
    plt.subplot(1, 3, 2)
    plt.imshow(processed.gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    # Normalized image
    plt.subplot(1, 3, 3)
    plt.imshow(processed.normalized_rgb)
    plt.title("Normalized RGB")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    sample_image = Path("dataset/real/real_001.jpeg")

    show_preprocessing(sample_image)