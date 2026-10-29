from fastapi import FastAPI

app = FastAPI(title="Anomaly Detection API")


@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/info")
def info():
    """Returns basic metadata about the API."""
    return {
        "project": "Anomaly Detection MLOps Pipeline",
        "version": "0.1.0",
        "description": "API for serving anomaly detection predictions",
    }
