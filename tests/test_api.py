from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    assert "project" in response.json()
