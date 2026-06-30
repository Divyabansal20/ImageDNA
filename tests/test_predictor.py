from pathlib import Path

from src.predictor import predict_image

REAL_DIR = Path("dataset/real")
SCREEN_DIR = Path("dataset/screen")


def evaluate(folder, label):

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    correct = 0
    total = 0

    confidences = []

    wrong = []

    for image in sorted(folder.glob("*.jpeg")):

        result = predict_image(image)

        prediction = result["prediction"]

        confidence = result["confidence"]

        confidences.append(confidence)

        total += 1

        if prediction == label:

            correct += 1

        else:

            wrong.append(
                (
                    image.name,
                    prediction,
                    confidence
                )
            )

        print(
            f"{image.name:18s}"
            f"{prediction:18s}"
            f"{confidence:6.2f}%"
        )

    print("\nAccuracy")

    print(f"{correct}/{total}")

    print(f"{correct/total*100:.2f}%")

    print("\nAverage Confidence")

    print(f"{sum(confidences)/len(confidences):.2f}%")

    print("\nWrong Predictions")

    for item in wrong:

        print(item)


evaluate(
    REAL_DIR,
    "Real Photo"
)

evaluate(
    SCREEN_DIR,
    "Screen Capture"
)