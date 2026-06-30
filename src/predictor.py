"""
predict.py

Predicts whether an image is a
real camera photo or a screen capture.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.config import MODELS_DIR
from src.preprocessing import preprocess_image
from src.feature_extractor import extract_features, FEATURE_NAMES


# Features used while training the model
TRAIN_FEATURES = [

    "sharpness",

    "fft_mean",
    "fft_std",
    "fft_energy",
    "high_frequency_ratio",

    "noise_mean",
    "noise_std",
    "noise_energy",

    "edge_density",

    "lbp_mean",
    "lbp_std",

    "contrast",
    "homogeneity",
    "correlation",

    "wavelet_mean",
    "wavelet_std",
    "wavelet_energy",

    "jpeg_horizontal",
    "jpeg_vertical",
    "jpeg_blocking_score",
]


def predict_image(image_path: Path):
    """
    Predicts whether an image is a
    real photo or a screen capture.
    """

    # Load trained model
    model_path = MODELS_DIR / "random_forest.pkl"
    model = joblib.load(model_path)

    print(model.feature_names_in_)

    # Preprocess image
    processed = preprocess_image(image_path)

    # Extract features and forensic evidence
    features, evidence = extract_features(
        processed,
        return_evidence=True
    )

    # Map features to FEATURE_NAMES
    features_dict = dict(zip(FEATURE_NAMES, features))

    # Keep only the features used during training
    feature_dataframe = pd.DataFrame(
        [[features_dict[name] for name in TRAIN_FEATURES]],
        columns=TRAIN_FEATURES
    )

    # Predict class
    prediction = model.predict(feature_dataframe)[0]

    # Prediction confidence
    probabilities = model.predict_proba(feature_dataframe)[0]

    # Borderline override: If the statistical model is borderline (Screen Capture probability >= 45%)
    # and explicit forensic anomalies (e.g. weak edge density or reduced frequency) are detected,
    # override to Screen Capture to align with explainable machine learning principles.
    if prediction == 0 and probabilities[1] >= 0.45:
        suspicious_evidence = {
            "Weak edge density",
            "Reduced high-frequency information",
            "Low image sharpness"
        }
        if any(item in suspicious_evidence for item in evidence):
            prediction = 1
            confidence = probabilities[1] * 100
        else:
            confidence = probabilities[0] * 100
    else:
        confidence = probabilities[prediction] * 100

    # Convert numeric prediction to text
    if prediction == 0:
        label = "Real Photo"
    else:
        label = "Screen Capture"

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "evidence": evidence,
    }


if __name__ == "__main__":

    # Change this path to test any image
    image = Path("dataset/real/real_001.jpeg")

    result = predict_image(image)

    print("\nPrediction")
    print("-" * 35)


    print(result["prediction"])

    confidence = result["confidence"]

    if confidence >= 90:
        level = "High Confidence"

    elif confidence >= 75:
        level = "Moderate Confidence"

    elif confidence >= 60:
        level = "Low Confidence"

    else:
        level = "Uncertain"

    print(f"\nConfidence : {confidence:.2f}% ({level})")

    print("\nTop Evidence")

    if result["evidence"]:

        for item in result["evidence"]:
            print(f"✓ {item}")

    else:

        print("✓ No strong forensic indicators detected.")