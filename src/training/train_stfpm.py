import json
from pathlib import Path

import torch
from anomalib.engine import Engine
from anomalib.models import Stfpm
import mlflow

from src.training.mlflow_utils import setup_mlflow

from src.training.datamodule import get_datamodule

torch.set_float32_matmul_precision("high")

RESULTS_DIR = Path("results/stfpm")


def train_stfpm(category: str, max_epochs: int = 50):
    """Train and evaluate STFPM on a given MVTec AD category."""
    datamodule = get_datamodule(category)
    model = Stfpm()

    engine = Engine(
        default_root_dir=str(RESULTS_DIR / category),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
    )

    with mlflow.start_run(run_name=f"STFPM_{category}"):
        mlflow.log_param("model", "STFPM")
        mlflow.log_param("category", category)

        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)

        if test_results:
            for key, value in test_results[0].items():
                mlflow.log_metric(key, value)

    return test_results


if __name__ == "__main__":
    setup_mlflow("STFPM-anomaly-detection")
    categories = ["bottle", "screw", "cable"]
    all_results = {}

    for category in categories:
        print(f"\n{'='*50}")
        print(f"Training STFPM on: {category}")
        print(f"{'='*50}\n")
        results = train_stfpm(category)
        all_results[category] = results
        print(f"Results for {category}: {results}")

    Path("results").mkdir(exist_ok=True)
    with open("results/stfpm_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\nSaved results to results/stfpm_summary.json")
