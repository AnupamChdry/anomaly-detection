import subprocess
import time

import pytest
import requests

IMAGE_NAME = "anomaly-detection-app:test"
CONTAINER_NAME = "anomaly-detection-test-container"
PORT = 8001  # different port to avoid clashing with local dev server

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def running_container():
    """Build and run the Docker container for the duration of these tests."""
    subprocess.run(["docker", "build", "-t", IMAGE_NAME, "."], check=True)

    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{PORT}:8000",
            IMAGE_NAME,
        ],
        check=True,
    )

    time.sleep(5)  # give the server a moment to start

    yield f"http://localhost:{PORT}"

    subprocess.run(["docker", "stop", CONTAINER_NAME], check=False)
    subprocess.run(["docker", "rm", CONTAINER_NAME], check=False)


def test_container_health_check(running_container):
    response = requests.get(f"{running_container}/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_container_predict_normal_image(running_container):
    with open("data/raw/bottle/test/good/000.png", "rb") as f:
        response = requests.post(
            f"{running_container}/predict?category=bottle",
            files={"file": ("000.png", f, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["pred_label"] in ["normal", "anomalous"]


def test_container_predict_anomalous_image(running_container):
    with open("data/raw/bottle/test/broken_large/000.png", "rb") as f:
        response = requests.post(
            f"{running_container}/predict?category=bottle",
            files={"file": ("000.png", f, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["pred_label"] == "anomalous"
