"""
trainer.py

Trains and evaluates the Random Forest classifier
using 5-Fold Cross Validation.
"""

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.metrics import confusion_matrix

from src.config import MODELS_DIR, OUTPUTS_DIR
from src.feature_extractor import FEATURE_NAMES

# Features that will actually be used for training
TRAIN_FEATURES = [

    "sharpness",

    "fft_mean",
    "fft_std",
    "fft_energy",

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


def train():
    """
    Trains the Random Forest model and evaluates it
    using 5-Fold Cross Validation.
    """

    csv_path = OUTPUTS_DIR / "features.csv"

    dataframe = pd.read_csv(csv_path)

    X = dataframe[TRAIN_FEATURES]
    y = dataframe["label"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=42,
    )

    print("Running 5-Fold Cross Validation...\n")

    kfold = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=kfold,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
        ],
    )

    print("Fold Accuracies\n")

    for i, score in enumerate(scores["test_accuracy"], start=1):
        print(f"Fold {i} : {score:.4f}")

    print("\nAverage Performance\n")

    print(f"Accuracy  : {scores['test_accuracy'].mean()*100:.2f}%")
    print(f"Precision : {scores['test_precision'].mean()*100:.2f}%")
    print(f"Recall    : {scores['test_recall'].mean()*100:.2f}%")
    print(f"F1 Score  : {scores['test_f1'].mean()*100:.2f}%")

    print(f"\nAccuracy Std Dev : {scores['test_accuracy'].std()*100:.2f}%")

    # --------------------------------------------------
    # Misclassification Report
    # --------------------------------------------------

    print("\nFinding misclassified images...\n")

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=kfold
    )

    cm = confusion_matrix(y, predictions)

    print("Confusion Matrix\n")
    print(cm)

    report = pd.DataFrame({
        "image_name": dataframe["image_name"],
        "Actual": y,
        "Predicted": predictions,
    })

    misclassified = report[
        report["Actual"] != report["Predicted"]
    ]

    print(f"\nTotal Misclassified Images : {len(misclassified)}")

    report_path = OUTPUTS_DIR / "misclassified_images.csv"

    misclassified.to_csv(
        report_path,
        index=False,
    )

    print(f"Report saved to : {report_path}")

    # --------------------------------------------------
    # Train final model
    # --------------------------------------------------

    model.fit(X, y)

    print(f"\nOut-of-Bag Score : {model.oob_score_ * 100:.2f}%")

    MODELS_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / "random_forest.pkl"

    joblib.dump(model, model_path)

    print(f"\nModel saved to : {model_path}")

    print("\nFeature Importance\n")

    importance = pd.DataFrame({
    "Feature": TRAIN_FEATURES,
    "Importance": model.feature_importances_,
})

    importance = importance.sort_values(
        by="Importance",
        ascending=False,
    )

    print(importance.to_string(index=False))


if __name__ == "__main__":
    train()