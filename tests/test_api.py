from fastapi.testclient import TestClient
from api.main import app
import pytest

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    assert "project" in response.json()


@pytest.mark.integration
def test_predict_valid_image():
    with open("data/raw/bottle/test/good/000.png", "rb") as f:
        response = client.post(
            "/predict?category=bottle",
            files={"file": ("000.png", f, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "pred_score" in data
    assert "pred_label" in data

@pytest.mark.integration
def test_predict_invalid_category():
    with open("data/raw/bottle/test/good/000.png", "rb") as f:
        response = client.post(
            "/predict?category=invalid_category",
            files={"file": ("000.png", f, "image/png")},
        )
    assert response.status_code == 400
