# Monitoring Setup

Due to a dependency conflict between evidently==0.4.25 and the newer
numpy 2.x / pandas 3.x versions required by torch/anomalib (used for
training and serving), report generation runs in an isolated virtual
environment with pinned older versions.

## One-time setup

    # Requires a Python version <= 3.12 (NumPy 1.26.4 has no prebuilt
    # wheels for 3.13+). If your default Python is newer, use pyenv:
    #   pyenv install 3.12.10
    #   "$PYENV_ROOT/versions/3.12.10/python.exe" -m venv monitoring-env

    python -m venv monitoring-env
    source monitoring-env/Scripts/activate
    pip install -r requirements-monitoring.txt
    deactivate

## Usage

    bash run_monitoring_report.sh --category all

Or manually:

    source monitoring-env/Scripts/activate
    python -m src.monitoring.generate_report --category all
    deactivate

Note: build_reference.py and log_prediction.py run fine in the main
environment (anomaly-env) since they don't depend on evidently --
only generate_report.py requires the isolated environment.

## Validated Drift Detection Experiment

To confirm the monitoring pipeline detects real distributional shifts
(not just running without error), a controlled degradation experiment
was performed:

1. 20 bottle test images were darkened to 40% brightness and blurred
   (Gaussian blur, radius=3) via src/monitoring/simulate_drift.py
2. These degraded images were sent through /predict
3. A fresh drift report was generated and compared against baseline

Result: `image_mean_brightness`, `image_sharpness`, and `anomaly_score`
were all correctly flagged as drifted, confirming:
- Input-statistic monitoring detects systemic image quality shifts
  (e.g. camera/lighting degradation), as designed
- The model's own anomaly scoring is also sensitive to degraded
  image quality, providing an independent corroborating signal
