import csv

from PIL import Image

from src.monitoring import log_prediction as log_prediction_module
from src.monitoring.log_prediction import log_prediction


def test_log_prediction_appends_row_with_correct_schema(tmp_path, monkeypatch):
    # Redirect the log output to a temp file so we don't pollute the real log
    temp_log_path = tmp_path / "predictions_log.csv"
    monkeypatch.setattr(log_prediction_module, "OUTPUT_PATH", temp_log_path)

    dummy_image = Image.new("RGB", (100, 100), color=(128, 128, 128))

    log_prediction(
        category="bottle",
        image=dummy_image,
        filename="test_image.png",
        anomaly_score=0.42,
        is_anomalous=False,
        model_used="patchcore",
    )

    assert temp_log_path.exists()

    with open(temp_log_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["category"] == "bottle"
    assert row["image_path"] == "test_image.png"
    assert row["anomaly_score"] == "0.42"
    assert row["is_anomalous"] == "False"
    assert row["model_used"] == "patchcore"


def test_log_prediction_appends_multiple_rows_without_duplicate_headers(
    tmp_path, monkeypatch
):
    temp_log_path = tmp_path / "predictions_log.csv"
    monkeypatch.setattr(log_prediction_module, "OUTPUT_PATH", temp_log_path)

    dummy_image = Image.new("RGB", (50, 50), color=(200, 200, 200))

    for _ in range(3):
        log_prediction(
            category="screw",
            image=dummy_image,
            filename="test.png",
            anomaly_score=0.1,
            is_anomalous=False,
        )

    with open(temp_log_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
