"""
log_prediction.py

Appends live /predict API results to a growing CSV log, using the same
schema as the reference datasets (see build_reference.py), so the two
can be directly compared via Evidently (see generate_report.py, Day 4).

This log is intentionally kept local and gitignored -- it is runtime
data, not a reproducible pipeline artifact. See README "Monitoring"
section for the Option B design decision and its tradeoffs.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.monitoring.image_stats import compute_image_stats_from_image

OUTPUT_PATH = Path("results/monitoring/predictions_log.csv")

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


def log_prediction(
    category: str,
    image: Image.Image,
    filename: str,
    anomaly_score: float,
    is_anomalous: bool,
    model_used: str = "patchcore",
) -> None:
    """
    Append a single prediction record to the live predictions log.

    Note: `filename` is the uploaded file's original filename only
    (not a real filesystem path), since uploaded images are never
    persisted to disk in this API. See Day 3 design notes.
    """
    stats = compute_image_stats_from_image(image)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "image_path": filename,
        "image_mean_brightness": stats["image_mean_brightness"],
        "image_std_brightness": stats["image_std_brightness"],
        "image_sharpness": stats["image_sharpness"],
        "image_width": stats["image_width"],
        "image_height": stats["image_height"],
        "anomaly_score": anomaly_score,
        "is_anomalous": is_anomalous,
        "model_used": model_used,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_PATH.exists()

    with open(OUTPUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
