from anomalib.data import MVTecAD

CATEGORIES = ["bottle", "screw", "cable"]
DATA_ROOT = "data/raw"


def get_datamodule(category: str, batch_size: int = 32) -> MVTecAD:
    """Create an anomalib MVTecAD datamodule for a given category."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Must be one of {CATEGORIES}")

    datamodule = MVTecAD(
        root=DATA_ROOT,
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
    )
    datamodule.setup()
    return datamodule


if __name__ == "__main__":
    for category in CATEGORIES:
        dm = get_datamodule(category)
        print(f"{category}: train={len(dm.train_data)}, test={len(dm.test_data)}")
