"""
Run inference on a single image using the trained skin lesion classifier.

Usage:
    python predict.py --image path/to/image.jpg --model skin_lesion_FINAL_best.pt
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Classify a skin lesion image.")
    parser.add_argument("--image", required=True, help="Path to the image to classify.")
    parser.add_argument("--model", default="skin_lesion_FINAL_best.pt", help="Path to the trained .pt model file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to display.")
    args = parser.parse_args()

    model = YOLO(args.model)
    result = model.predict(args.image, verbose=False)[0]

    probs = result.probs
    class_names = result.names

    top_indices = probs.top5[: args.topk]
    print(f"\nPredictions for: {args.image}\n" + "-" * 40)
    for idx in top_indices:
        name = class_names[idx]
        confidence = float(probs.data[idx]) * 100
        print(f"{name:35s} {confidence:6.2f}%")

    print("\nNote: this model is for educational/demo purposes only and is not")
    print("validated for clinical use. Do not use for real medical decisions.")


if __name__ == "__main__":
    main()
