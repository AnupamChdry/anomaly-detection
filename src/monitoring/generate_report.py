"""
generate_report.py

Generates an Evidently AI drift report comparing a category's reference
dataset (baseline statistics from training data) against its live
predictions log (real /predict API calls), producing an HTML report.

Usage:
    python -m src.monitoring.generate_report --category bottle
    python -m src.monitoring.generate_report --category all

NOTE: This script must be run inside the isolated `monitoring-env`
virtual environment (not the main `anomaly-env`), due to a dependency
conflict between evidently==0.4.25 and the newer numpy/pandas versions
required by torch/anomalib in the main training/serving environment.
See README "Monitoring" section for details.

    source monitoring-env/Scripts/activate
    python -m src.monitoring.generate_report --category all
    deactivate

Note: statistical drift tests are only meaningful with a reasonable
sample size in the "current" dataset. With very few logged predictions,
results should be treated as illustrative, not statistically reliable.
"""

from datetime import datetime
import argparse
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = PROJECT_ROOT / "results" / "monitoring"
REPORTS_DIR = MONITORING_DIR / "reports"

CATEGORIES = ["bottle", "screw", "cable"]

NUMERICAL_FEATURES = [
    "image_mean_brightness",
    "image_std_brightness",
    "image_sharpness",
    "image_width",
    "image_height",
    "anomaly_score",
]
CATEGORICAL_FEATURES = ["is_anomalous", "model_used"]


def load_reference(category: str) -> pd.DataFrame:
    ref_path = MONITORING_DIR / f"reference_{category}.csv"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {ref_path}. "
            f"Run: python -m src.monitoring.build_reference --category {category}"
        )
    return pd.read_csv(ref_path, parse_dates=["timestamp"])


def load_current(category: str) -> pd.DataFrame:
    log_path = MONITORING_DIR / "predictions_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(
            f"Predictions log not found at {log_path}. "
            f"Call /predict at least once (locally or via Docker) before "
            f"generating a report."
        )
    df = pd.read_csv(log_path, parse_dates=["timestamp"])
    df = df[df["category"] == category].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No logged predictions found for category '{category}' in "
            f"{log_path}. Call /predict with category='{category}' first."
        )
    return df


def generate_report_for_category(category: str) -> Path:
    print(f"[{category}] Loading reference and current data...")
    reference_df = load_reference(category)
    current_df = load_current(category)

    print(
        f"[{category}] Reference rows: {len(reference_df)}, "
        f"Current rows: {len(current_df)}"
    )
    if len(current_df) < 30:
        print(
            f"[{category}] WARNING: only {len(current_df)} current rows. "
            f"Drift test results will be statistically weak with such a "
            f"small sample."
        )

    column_mapping = ColumnMapping(
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        datetime="timestamp",
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = REPORTS_DIR / f"drift_report_{category}_{timestamp}.html"
    report.save_html(str(output_path))

    # Also save a stable "latest" copy for convenience
    latest_path = REPORTS_DIR / f"drift_report_{category}_latest.html"
    report.save_html(str(latest_path))

    print(f"[{category}] Report saved to {output_path}")
    print(f"[{category}] Latest copy saved to {latest_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Evidently drift reports for drift monitoring."
    )
    parser.add_argument(
        "--category",
        choices=CATEGORIES + ["all"],
        default="all",
        help="Category to generate a report for, or 'all'. Default: all.",
    )
    args = parser.parse_args()

    categories = CATEGORIES if args.category == "all" else [args.category]

    for category in categories:
        try:
            generate_report_for_category(category)
        except (FileNotFoundError, ValueError) as e:
            print(f"[{category}] Skipped: {e}")


if __name__ == "__main__":
    main()
