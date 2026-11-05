from pathlib import Path
import torch

from anomalib.engine import Engine
from anomalib.models import Patchcore
import mlflow

from src.training.mlflow_utils import setup_mlflow
from src.training.datamodule import get_datamodule

RESULTS_DIR = Path("results/patchcore")


def train_patchcore(category: str):
    """Train and evaluate PatchCore on a given MVTec AD category."""
    datamodule = get_datamodule(category)
    model = Patchcore()

    engine = Engine(
        default_root_dir=str(RESULTS_DIR / category),
    )

    with mlflow.start_run(run_name=f"patchcore_{category}"):
        mlflow.log_param("model", "PatchCore")
        mlflow.log_param("category", category)

        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)

        if test_results:
            for key, value in test_results[0].items():
                mlflow.log_metric(key, value)

    return test_results


if __name__ == "__main__":
    setup_mlflow("patchcore-anomaly-detection")
    categories = ["bottle", "screw", "cable"]
    all_results = {}

    for category in categories:
        print(f"\n{'='*50}")
        print(f"Training PatchCore on: {category}")
        print(f"{'='*50}\n")
        results = train_patchcore(category)
        all_results[category] = results
        print(f"Results for {category}: {results}")

    print("\n\nSummary of all results:")
    for category, results in all_results.items():
        print(f"{category}: {results}")

    import json

    with open("results/patchcore_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
