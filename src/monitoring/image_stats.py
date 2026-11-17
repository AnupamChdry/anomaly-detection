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


def compute_image_stats(image_path: str) -> dict:
    """
    Compute lightweight statistical summaries of an image.

    Args:
        image_path: path to an image file on disk.

    Returns:
        dict with keys: mean_brightness, std_brightness, sharpness,
        width, height.
    """
    # Load as grayscale for brightness/contrast/sharpness calculations
    img_gray = Image.open(image_path).convert("L")
    img_array = np.array(img_gray)

    width, height = img_gray.size

    mean_brightness = float(np.mean(img_array))
    std_brightness = float(np.std(img_array))

    # Laplacian variance: higher = sharper/more in-focus, lower = blurrier
    sharpness = float(cv2.Laplacian(img_array, cv2.CV_64F).var())

    return {
        "image_mean_brightness": mean_brightness,
        "image_std_brightness": std_brightness,
        "image_sharpness": sharpness,
        "image_width": width,
        "image_height": height,
    }
