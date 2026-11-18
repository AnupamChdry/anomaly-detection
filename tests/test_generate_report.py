"""
test_generate_report.py

Tests for Evidently drift report generation.

NOTE: This test requires the `evidently` package, which is only
installed in the isolated `monitoring-env` virtual environment (not
the main `anomaly-env`), due to a dependency conflict between
evidently==0.4.25 and the newer numpy/pandas versions required by
torch/anomalib. See src/monitoring/README.md for details.

When run from the main test suite (anomaly-env) or CI, this test is
SKIPPED automatically. To actually run it:

    source monitoring-env/Scripts/activate
    python -m pytest tests/test_generate_report.py -v
    deactivate
"""

import pytest

try:
    from src.monitoring.generate_report import generate_report_for_category
except ModuleNotFoundError as e:
    pytest.skip(f"Skipping drift report tests: {e}", allow_module_level=True)


@pytest.mark.integration
def test_generate_report_for_bottle_creates_html():
    output_path = generate_report_for_category("bottle")

    assert output_path.exists()
    assert output_path.suffix == ".html"
    assert output_path.stat().st_size > 0
    assert "drift_report_bottle" in output_path.name