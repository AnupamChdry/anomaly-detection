from pathlib import Path

import torch
from anomalib.models import Patchcore
from PIL import Image
from torchvision.transforms import v2

CHECKPOINT_DIR = Path("results/patchcore")


def find_latest_checkpoint(category: str) -> Path:
    """Find the most recent PatchCore checkpoint for a given category."""
    category_dir = CHECKPOINT_DIR / category / "Patchcore" / "MVTecAD" / category

    # Prefer the 'latest' symlink/folder if it exists (used in CI/production)
    latest_path = category_dir / "latest" / "weights" / "lightning" / "model.ckpt"
    if latest_path.exists():
        return latest_path

    # Fall back to searching versioned folders (used in local training)
    version_dirs = sorted(
        [d for d in category_dir.glob("v*") if d.is_dir()],
        key=lambda d: int(d.name[1:]),
    )
    if not version_dirs:
        raise FileNotFoundError(f"No checkpoint found for category: {category}")

    checkpoint_path = version_dirs[-1] / "weights" / "lightning" / "model.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    return checkpoint_path


def load_model(category: str) -> Patchcore:
    """Load a trained PatchCore model for a given category."""
    checkpoint_path = find_latest_checkpoint(category)
    model = Patchcore.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    return model


def preprocess_image(
    image: Image.Image, image_size: tuple = (256, 256)
) -> torch.Tensor:
    """Preprocess a PIL image into a model-ready tensor."""
    transform = v2.Compose(
        [
            v2.Resize(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    tensor = transform(image)
    return tensor.unsqueeze(0)


def predict(model: Patchcore, image: Image.Image) -> dict:
    """Run anomaly detection prediction on a single image."""
    device = next(model.parameters()).device
    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    pred_score = float(output.pred_score.item())
    pred_label = "anomalous" if output.pred_label.item() else "normal"

    return {
        "pred_score": pred_score,
        "pred_label": pred_label,
    }


if __name__ == "__main__":
    import sys

    category = sys.argv[1] if len(sys.argv) > 1 else "bottle"
    test_image_path = (
        sys.argv[2] if len(sys.argv) > 2 else "data/raw/bottle/test/good/000.png"
    )

    model = load_model(category)
    image = Image.open(test_image_path).convert("RGB")
    result = predict(model, image)
    print(result)
