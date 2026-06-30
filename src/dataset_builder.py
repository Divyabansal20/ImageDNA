"""
dataset_builder.py

Builds the training dataset by extracting
features from every image.
"""

from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_DIR, REAL_DIR, SCREEN_DIR
from src.feature_extractor import FEATURE_NAMES, extract_features
from src.preprocessing import preprocess_image


def process_folder(folder: Path, label: int):
    """
    Extracts features from every image inside one folder.
    """

    dataset = []

    for image_path in sorted(folder.glob("*")):

        processed = preprocess_image(image_path)

        features = extract_features(processed)

        # Store one complete row
        row = {
            "image_name": image_path.name
        }

        # Add every extracted feature
        for feature_name, value in zip(FEATURE_NAMES, features):
            row[feature_name] = value

        # Store class label
        row["label"] = label

        dataset.append(row)

    return dataset


def build_dataset():

    print("Extracting features from real images...")

    real_data = process_folder(REAL_DIR, 0)

    print("Extracting features from screen images...")

    screen_data = process_folder(SCREEN_DIR, 1)

    dataset = real_data + screen_data

    dataframe = pd.DataFrame(dataset)

    OUTPUTS_DIR.mkdir(exist_ok=True)

    save_path = OUTPUTS_DIR / "features.csv"

    dataframe.to_csv(save_path, index=False)

    print("\nDataset created successfully!")

    print(f"Saved to : {save_path}")

    print(f"Total Samples : {len(dataframe)}")

    print(dataframe.head())


if __name__ == "__main__":

    build_dataset()