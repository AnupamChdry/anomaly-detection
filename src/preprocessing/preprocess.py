import os
import cv2
import numpy as np
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
TARGET_SIZE = (256, 256)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk using OpenCV."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def resize_image(image: np.ndarray, size: tuple = TARGET_SIZE) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to range [0, 1]."""
    return image.astype(np.float32) / 255.0


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save a normalized image back to disk as 8-bit PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert back to 0-255 range for saving as a standard image file
    image_to_save = (image * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), image_to_save)


def process_image(input_path: Path, output_path: Path) -> None:
    """Full pipeline for a single image: load, resize, normalize, save."""
    image = load_image(input_path)
    image = resize_image(image)
    image = normalize_image(image)
    save_image(image, output_path)


def process_dataset(raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Walk through raw dataset, process every image, mirror folder structure."""
    image_extensions = {".png", ".jpg", ".jpeg"}

    for raw_path in raw_dir.rglob("*"):
        if raw_path.suffix.lower() in image_extensions:
            relative_path = raw_path.relative_to(raw_dir)
            output_path = processed_dir / relative_path
            process_image(raw_path, output_path)
            print(f"Processed: {relative_path}")


if __name__ == "__main__":
    process_dataset()
    print("Preprocessing complete.")