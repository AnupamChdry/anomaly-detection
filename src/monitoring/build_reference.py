"""
build_reference.py

Builds reference datasets for drift monitoring by computing image
statistics and model predictions on the "good" (defect-free) training
images for each MVTec AD category. These serve as the baseline
distribution that live predictions (see log_prediction.py) are later
compared against.

Usage:
    python -m src.monitoring.build_reference
    python -m src.monitoring.build_reference --category bottle
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.inference.predict import load_model, predict
from src.monitoring.image_stats import compute_image_stats

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("results/monitoring")
CATEGORIES = ["bottle", "screw", "cable"]
MODEL_USED = "patchcore"

FIELDNAMES = [
    "timestamp",
    "category",
    "image_path",
    "image_mean_brightness",
    "image_std_brightness",
    "image_sharpness",
    "image_width",
    "image_height",
    "anomaly_score",
    "is_anomalous",
    "model_used",
]


def build_reference_for_category(category: str) -> Path:
    """Build a reference CSV for a single category using its training images."""
    train_dir = DATA_DIR / category / "train" / "good"
    image_paths = sorted(train_dir.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No training images found at {train_dir}")

    print(f"[{category}] Loading model...")
    model = load_model(category)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"reference_{category}.csv"

    print(f"[{category}] Processing {len(image_paths)} training images...")
    rows = []
    for image_path in image_paths:
        stats = compute_image_stats(str(image_path))

        image = Image.open(image_path).convert("RGB")
        result = predict(model, image)

        rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "category": category,
                "image_path": image_path.relative_to(DATA_DIR).as_posix(),
                "image_mean_brightness": stats["image_mean_brightness"],
                "image_std_brightness": stats["image_std_brightness"],
                "image_sharpness": stats["image_sharpness"],
                "image_width": stats["image_width"],
                "image_height": stats["image_height"],
                "anomaly_score": result["pred_score"],
                "is_anomalous": result["pred_label"] == "anomalous",
                "model_used": MODEL_USED,
            }
        )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{category}] Wrote {len(rows)} rows to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build reference datasets for drift monitoring."
    )
    parser.add_argument(
        "--category",
        choices=CATEGORIES,
        default=None,
        help="Build reference for a single category only. Default: all categories.",
    )
    args = parser.parse_args()

    categories = [args.category] if args.category else CATEGORIES

    for category in categories:
        build_reference_for_category(category)


if __name__ == "__main__":
    main()
