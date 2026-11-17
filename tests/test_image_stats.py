import glob
import pytest
from src.monitoring.image_stats import compute_image_stats


@pytest.mark.integration
def test_compute_image_stats_returns_expected_keys():
    sample_images = glob.glob("data/raw/bottle/test/*/*.png")
    assert len(sample_images) > 0, "No sample images found for testing"

    result = compute_image_stats(sample_images[0])

    expected_keys = {
        "image_mean_brightness",
        "image_std_brightness",
        "image_sharpness",
        "image_width",
        "image_height",
    }
    assert set(result.keys()) == expected_keys

    assert 0 <= result["image_mean_brightness"] <= 255
    assert result["image_std_brightness"] >= 0
    assert result["image_sharpness"] >= 0
    assert result["image_width"] > 0
    assert result["image_height"] > 0
    