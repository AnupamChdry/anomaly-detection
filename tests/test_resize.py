import numpy as np
from src.preprocessing.resize import resize_image


def test_resize_image():
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    resized = resize_image(dummy_image, 50, 50)
    assert resized.shape == (50, 50, 3)
    