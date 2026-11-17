import csv

import pytest

from src.monitoring.build_reference import build_reference_for_category, FIELDNAMES


@pytest.mark.integration
def test_build_reference_for_bottle_creates_valid_csv():
    output_path = build_reference_for_category("bottle")

    assert output_path.exists()

    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    assert set(reader.fieldnames) == set(FIELDNAMES)

    first_row = rows[0]
    first_row = rows[0]
    assert first_row["category"] == "bottle"
    assert first_row["image_path"].startswith("bottle/")
    assert first_row["model_used"] == "patchcore"
    assert first_row["is_anomalous"] in ("True", "False")
