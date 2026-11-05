import json
from pathlib import Path

from anomalib.engine import Engine
from anomalib.models import Stfpm

from src.training.datamodule import get_datamodule

RESULTS_DIR = Path("results/stfpm")


def find_latest_checkpoint(category: str) -> Path:
    """Find the most recent checkpoint for a given category."""
    category_dir = RESULTS_DIR / category / "Stfpm" / "MVTecAD" / category

    version_dirs = sorted(
        [d for d in category_dir.glob("v*") if d.is_dir()],
        key=lambda d: int(d.name[1:]),
    )

    if not version_dirs:
        raise FileNotFoundError(f"No version folders found for {category} in {category_dir}")

    latest_version_dir = version_dirs[-1]
    checkpoint_path = latest_version_dir / "weights" / "lightning" / "model.ckpt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    return checkpoint_path


def evaluate_from_checkpoint(category: str):
    """Load a trained STFPM checkpoint and re-run evaluation only."""
    checkpoint_path = find_latest_checkpoint(category)
    print(f"Loading checkpoint: {checkpoint_path}")

    datamodule = get_datamodule(category)
    model = Stfpm.load_from_checkpoint(str(checkpoint_path))

    engine = Engine(default_root_dir=str(RESULTS_DIR / category))
    results = engine.test(model=model, datamodule=datamodule)

    return results


if __name__ == "__main__":
    categories = ["bottle", "screw", "cable"]
    all_results = {}

    for category in categories:
        try:
            results = evaluate_from_checkpoint(category)
            all_results[category] = results
            print(f"{category}: {results}")
        except FileNotFoundError as e:
            print(f"Skipping {category}: {e}")

    Path("results").mkdir(exist_ok=True)
    with open("results/stfpm_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nSaved results to results/stfpm_summary.json")