import pytest
from PIL import Image

from src.inference.predict import load_model, predict

pytestmark = pytest.mark.integration


def test_predict_on_normal_image():
    model = load_model("bottle")
    image = Image.open("data/raw/bottle/test/good/000.png").convert("RGB")
    result = predict(model, image)
    assert "pred_score" in result
    assert "pred_label" in result
    assert result["pred_label"] in ["normal", "anomalous"]
