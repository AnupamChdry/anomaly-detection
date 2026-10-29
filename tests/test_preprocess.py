import numpy as np
from src.preprocessing.preprocess import resize_image, normalize_image


def test_resize_image():
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    resized = resize_image(dummy_image, (256, 256))
    assert resized.shape == (256, 256, 3)


def test_normalize_image():
    dummy_image = np.full((10, 10, 3), 255, dtype=np.uint8)
    normalized = normalize_image(dummy_image)
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0
    assert normalized.dtype == np.float32