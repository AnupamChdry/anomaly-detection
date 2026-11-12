import os

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.inference.predict import load_model, predict

app = FastAPI(title="Anomaly Detection API")

VALID_CATEGORIES = {"bottle", "screw", "cable"}

# Cache loaded models so we don't reload from disk on every request
_model_cache = {}


def get_model(category: str):
    if category not in _model_cache:
        _model_cache[category] = load_model(category)
    return _model_cache[category]


@app.get("/")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.get("/info")
def info():
    """Returns basic metadata about the API."""
    return {
        "project": "Anomaly Detection MLOps Pipeline",
        "version": "0.1.0",
        "model_version": MODEL_VERSION,
        "description": "API for serving anomaly detection predictions",
        "supported_categories": list(VALID_CATEGORIES),
    }


@app.post("/predict")
async def predict_endpoint(category: str, file: UploadFile = File(...)):
    """Run anomaly detection on an uploaded image for a given category."""
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Must be one of {VALID_CATEGORIES}",
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        model = get_model(category)
        result = predict(model, image)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"No trained model found for category '{category}'"
        )

    return {
        "category": category,
        "filename": file.filename,
        **result,
    }
