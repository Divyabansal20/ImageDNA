"""
feature_extractor.py

Extracts forensic features from an image.

Every function computes one independent forensic clue.
All extracted features are finally combined into a single
feature vector for the Random Forest classifier.
"""

import cv2
import numpy as np
from skimage.measure import shannon_entropy
import pywt

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)

from src.preprocessing import ProcessedImage

# Feature Names
FEATURE_NAMES = [

    "sharpness",

    "fft_mean",
    "fft_std",
    "fft_energy",
    "high_frequency_ratio",

    "noise_mean",
    "noise_std",
    "noise_energy",

    "edge_density",

    "entropy",

    "mean_r",
    "mean_g",
    "mean_b",

    "std_r",
    "std_g",
    "std_b",

    # LBP
    "lbp_mean",
    "lbp_std",

    # GLCM
    "contrast",
    "homogeneity",
    "energy",
    "correlation",

        # JPEG Blocking
    "jpeg_horizontal",
    "jpeg_vertical",
    "jpeg_blocking_score",



    # Wavelet
    "wavelet_mean",
    "wavelet_std",
    "wavelet_energy",

    # # Multi-scale
    # "laplacian_100",
    # "laplacian_50",
    # "laplacian_25"
]

# Sharpness

def extract_sharpness(processed: ProcessedImage) -> float:
    """
    Measures image sharpness using the Variance of Laplacian.

    A sharper image usually has a larger variance.
    """

    laplacian = cv2.Laplacian(processed.gray, cv2.CV_64F)

    sharpness = laplacian.var()

    return float(sharpness)

# FFT Features

def extract_fft_features(processed: ProcessedImage):
    """
    Extracts frequency-domain statistics.

    Besides overall FFT statistics, we also compute
    how much energy lies in the outer (high-frequency)
    region of the spectrum.
    """

    # Compute FFT
    fft = np.fft.fft2(processed.gray)

    fft_shift = np.fft.fftshift(fft)

    magnitude = np.log1p(np.abs(fft_shift))

    fft_mean = magnitude.mean()

    fft_std = magnitude.std()

    fft_energy = np.mean(magnitude ** 2)

    rows, cols = magnitude.shape

    center_row = rows // 2

    center_col = cols // 2

    radius = min(rows, cols) // 4

    y, x = np.ogrid[:rows, :cols]

    distance = np.sqrt(
        (x - center_col) ** 2 +
        (y - center_row) ** 2
    )

    high_frequency = magnitude[distance > radius]

    total_energy = np.sum(magnitude)

    high_energy = np.sum(high_frequency)

    high_frequency_ratio = high_energy / (total_energy + 1e-8)

    return (

        float(fft_mean),

        float(fft_std),

        float(fft_energy),

        float(high_frequency_ratio)

    )


# Noise Features

def extract_noise_features(processed: ProcessedImage):
    """
    Estimates image noise.

    The image is blurred and then subtracted
    from the original to obtain the noise residual.
    """

    blurred = cv2.GaussianBlur(
        processed.gray,
        (5, 5),
        0
    )

    noise = cv2.absdiff(
        processed.gray,
        blurred
    )

    noise_mean = noise.mean()

    noise_std = noise.std()

    # Average noise energy per pixel.
    noise_energy = np.mean(noise.astype(np.float32) ** 2)

    return (

        float(noise_mean),

        float(noise_std),

        float(noise_energy)

    )

# Edge Density

def extract_edge_density(processed: ProcessedImage):
    """
    Computes the percentage of edge pixels.

    Screen photos often lose tiny edges.
    """

    edges = cv2.Canny(

        processed.gray,

        100,

        200

    )

    edge_pixels = np.count_nonzero(edges)

    total_pixels = edges.size

    density = edge_pixels / total_pixels

    return float(density)

# Entropy

def extract_entropy(processed: ProcessedImage):
    """
    Measures information content.

    Images that lose detail generally
    have lower entropy.
    """

    entropy = shannon_entropy(processed.gray)

    return float(entropy)

# Color Statistics

def extract_color_features(processed: ProcessedImage):
    """
    Computes simple RGB statistics.

    Useful for detecting subtle
    color shifts introduced by displays.
    """

    rgb = processed.rgb

    mean_r = rgb[:, :, 0].mean()
    mean_g = rgb[:, :, 1].mean()
    mean_b = rgb[:, :, 2].mean()

    std_r = rgb[:, :, 0].std()
    std_g = rgb[:, :, 1].std()
    std_b = rgb[:, :, 2].std()

    return (

        float(mean_r),

        float(mean_g),

        float(mean_b),

        float(std_r),

        float(std_g),

        float(std_b)

    )

# -------------------------------------------------------
# Local Binary Pattern
# -------------------------------------------------------

def extract_lbp_features(processed: ProcessedImage):
    """
    Captures tiny local texture patterns.

    Screen recaptured images often introduce
    different microscopic textures compared
    to genuine photographs.
    """

    radius = 2
    neighbors = 8 * radius

    lbp = local_binary_pattern(
        processed.gray,
        P=neighbors,
        R=radius,
        method="uniform"
    )

    lbp_mean = lbp.mean()
    lbp_std = lbp.std()

    return (
        float(lbp_mean),
        float(lbp_std)
    )


# -------------------------------------------------------
# GLCM Texture
# -------------------------------------------------------

def extract_glcm_features(processed: ProcessedImage):
    """
    Measures relationships between neighboring pixels.

    Useful for detecting subtle texture differences.
    """

    image = processed.gray.astype(np.uint8)

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        symmetric=True,
        normed=True,
    )

    contrast = graycoprops(glcm, "contrast")[0, 0]

    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    energy = graycoprops(glcm, "energy")[0, 0]

    correlation = graycoprops(glcm, "correlation")[0, 0]

    return (
        float(contrast),
        float(homogeneity),
        float(energy),
        float(correlation),
    )


# -------------------------------------------------------
# JPEG Blocking Features
# -------------------------------------------------------

def extract_jpeg_features(processed: ProcessedImage):
    """
    Measures JPEG blocking artifacts.

    JPEG compresses images in 8×8 blocks.
    Double compression often strengthens
    these block boundaries.
    """

    image = processed.gray.astype(np.float32)

    horizontal_diff = []
    vertical_diff = []

    # Horizontal block boundaries
    for row in range(8, image.shape[0], 8):

        diff = np.abs(image[row, :] - image[row - 1, :])

        horizontal_diff.append(diff.mean())

    # Vertical block boundaries
    for col in range(8, image.shape[1], 8):

        diff = np.abs(image[:, col] - image[:, col - 1])

        vertical_diff.append(diff.mean())

    horizontal_score = np.mean(horizontal_diff)

    vertical_score = np.mean(vertical_diff)

    blocking_score = (horizontal_score + vertical_score) / 2

    return (

        float(horizontal_score),

        float(vertical_score),

        float(blocking_score)

    )


# -------------------------------------------------------
# Wavelet Features
# -------------------------------------------------------

def extract_wavelet_features(processed: ProcessedImage):
    """
    Extracts wavelet statistics.

    Wavelets measure image information at
    multiple resolutions.

    Images captured through another display
    usually lose fine wavelet details.
    """

    coeffs = pywt.dwt2(processed.gray, "haar")
    
    LL, (LH, HL, HH) = coeffs

    detail = np.concatenate([

        LH.flatten(),

        HL.flatten(),

        HH.flatten()

    ])

    wavelet_mean = detail.mean()

    wavelet_std = detail.std()

    wavelet_energy = np.mean(detail ** 2)

    return (

        float(wavelet_mean),

        float(wavelet_std),

        float(wavelet_energy)

    )

# -------------------------------------------------------
# Master Feature Extractor
# -------------------------------------------------------

def extract_features(processed: ProcessedImage, return_evidence=False):
    """
    Extracts all forensic features from an image.

    Returns
    -------
    features : list
        Numerical feature vector used for training.

    evidence : list (optional)
        Human-readable reasons that explain the prediction.
    """

    features = []

    evidence = []

    # -----------------------------
    # Sharpness
    # -----------------------------
    sharpness = extract_sharpness(processed)

    features.append(sharpness)

    if sharpness < 120:
        evidence.append("Low image sharpness")


    # -----------------------------
    # FFT


    fft_mean, fft_std, fft_energy, high_frequency_ratio = extract_fft_features(processed)

    features.extend([
        fft_mean,
        fft_std,
        fft_energy,
        high_frequency_ratio
    ])

    if high_frequency_ratio < 0.74:
        evidence.append("Reduced high-frequency information")

    if fft_energy < 5e8:
        evidence.append("Reduced high-frequency information")


    # -----------------------------
    # Noise
    # -----------------------------
    noise_mean, noise_std, noise_energy = extract_noise_features(processed)

    features.extend([

        noise_mean,

        noise_std,

        noise_energy

    ])

    if noise_std > 15:
        evidence.append("High sensor noise variation")


    # -----------------------------
    # Edge Density
    # -----------------------------
    edge_density = extract_edge_density(processed)

    features.append(edge_density)

    if edge_density < 0.08:
        evidence.append("Weak edge density")


    # -----------------------------
    # Entropy
    # -----------------------------
    entropy = extract_entropy(processed)

    features.append(entropy)

    if entropy < 6:
        evidence.append("Information loss detected")


    # -----------------------------
    # Color Statistics
    # -----------------------------
    color_features = extract_color_features(processed)

    features.extend(color_features)

        # -----------------------------
    # Local Binary Pattern
    # -----------------------------
    lbp_features = extract_lbp_features(processed)

    features.extend(lbp_features)

    # -----------------------------
    # GLCM Texture
    # -----------------------------
    glcm_features = extract_glcm_features(processed)

    features.extend(glcm_features)



    # JPEG Blocking
    
    jpeg_features = extract_jpeg_features(processed)

    features.extend(jpeg_features)

    # wavelet features

    wavelet_features = extract_wavelet_features(processed)

    features.extend(wavelet_features)



    if return_evidence:

        return features, evidence

    return features

    # -----------------------------


# -------------------------------------------------------
# Demo
# -------------------------------------------------------

if __name__ == "__main__":

    from pathlib import Path

    from src.preprocessing import preprocess_image

    image_path = Path("dataset/real/real_001.jpeg")

    processed = preprocess_image(image_path)

    features, evidence = extract_features(

        processed,

        return_evidence=True

    )

    print("\nExtracted Features\n")

    for name, value in zip(FEATURE_NAMES, features):

        print(f"{name:20s}: {value:.4f}")

    print("\nEvidence\n")

    if evidence:

        for item in evidence:

            print(f"• {item}")

    else:

        print("No suspicious forensic evidence detected.")