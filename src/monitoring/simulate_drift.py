"""
simulate_drift.py

Generates deliberately degraded copies of test images (darkened and/or
blurred) to simulate a real-world camera/lighting drift scenario, for
validating that the monitoring pipeline correctly detects engineered
distribution shifts.

Usage:
    python -m src.monitoring.simulate_drift --category bottle --count 20
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "monitoring" / "simulated_drift"

CATEGORIES = ["bottle", "screw", "cable"]


def degrade_image(image: Image.Image) -> Image.Image:
    """Darken and blur an image to simulate camera/lighting degradation."""
    darkened = ImageEnhance.Brightness(image).enhance(0.4)  # 40% of original brightness
    blurred = darkened.filter(ImageFilter.GaussianBlur(radius=3))
    return blurred


def generate_degraded_images(category: str, count: int) -> list[Path]:
    source_dir = DATA_DIR / category / "test"
    source_images = list(source_dir.glob("*/*.png"))

    if not source_images:
        raise FileNotFoundError(f"No test images found for category: {category}")

    random.shuffle(source_images)
    selected = source_images[:count]

    output_dir = OUTPUT_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for i, src_path in enumerate(selected):
        image = Image.open(src_path).convert("RGB")
        degraded = degrade_image(image)

        output_path = output_dir / f"degraded_{i:03d}.png"
        degraded.save(output_path)
        output_paths.append(output_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate degraded images to simulate drift."
    )
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    paths = generate_degraded_images(args.category, args.count)
    print(f"Generated {len(paths)} degraded images in {OUTPUT_DIR / args.category}")


if __name__ == "__main__":
    main()