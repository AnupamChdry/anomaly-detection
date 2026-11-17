"""
image_stats.py

Lightweight image-level statistics used as proxy signals for input
distribution drift monitoring. These are NOT defect detectors -- they
are coarse, aggregate summaries intended to catch systemic shifts in
image capture conditions (lighting, exposure, focus) over time.

Known limitation: localized changes (e.g. small defects) can leave
these aggregate statistics unchanged. See README "Monitoring" section
for full discussion.
"""

import numpy as np
import cv2
from PIL import Image


def compute_image_stats_from_image(image: Image.Image) -> dict:
    """
    Compute lightweight statistical summaries from an in-memory PIL image.

    Args:
        image: a PIL Image object (any mode; will be converted to grayscale).

    Returns:
        dict with keys: image_mean_brightness, image_std_brightness,
        image_sharpness, image_width, image_height.
    """
    img_gray = image.convert("L")
    img_array = np.array(img_gray)

    width, height = img_gray.size

    mean_brightness = float(np.mean(img_array))
    std_brightness = float(np.std(img_array))
    sharpness = float(cv2.Laplacian(img_array, cv2.CV_64F).var())

    return {
        "image_mean_brightness": mean_brightness,
        "image_std_brightness": std_brightness,
        "image_sharpness": sharpness,
        "image_width": width,
        "image_height": height,
    }


def compute_image_stats(image_path: str) -> dict:
    """
    Compute lightweight statistical summaries of an image from disk.

    Args:
        image_path: path to an image file on disk.

    Returns:
        dict with keys: image_mean_brightness, image_std_brightness,
        image_sharpness, image_width, image_height.
    """
    image = Image.open(image_path).convert("RGB")
    return compute_image_stats_from_image(image)
