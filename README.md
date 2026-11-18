## Getting Started

### Prerequisites
- Python 3.10+
- Docker
- Git
- A DagsHub account (free) if you want to pull the trained model checkpoints via DVC

### 1. Clone the Repository

    git clone https://github.com/AnupamChdry/anomaly-detection.git
    cd anomaly-detection

### 2. Set Up the Main Environment

    python -m venv anomaly-env
    source anomaly-env/Scripts/activate   # Windows Git Bash
    # source anomaly-env/bin/activate     # Linux/Mac

    pip install -r requirements-dev.txt

### 3. Pull Data and Model Checkpoints (DVC)

This project uses DVC with a DagsHub remote to version the MVTec AD
dataset and trained model checkpoints (not stored in Git directly).

    dvc remote modify origin --local auth basic
    dvc remote modify origin --local user <your-dagshub-username>
    dvc remote modify origin --local password <your-dagshub-token>

    dvc pull

> Note: `data/raw` (~821MB) and `results/` (model checkpoints) are
> pulled from DagsHub storage. If you only want to run the API without
> retraining, you can pull selectively:
>
>     dvc pull results/patchcore/bottle/...
>     dvc pull results/patchcore/screw/...
>     dvc pull results/patchcore/cable/...

### 4. Run the API Locally

    uvicorn api.main:app --reload

Then test it:

    curl http://localhost:8000/
    curl http://localhost:8000/info

    curl -X POST "http://localhost:8000/predict?category=bottle" \
      -F "file=@data/raw/bottle/test/good/000.png"

### 5. Run via Docker (Production-Equivalent)

    docker build -t anomaly-detection .
    docker run -p 8000:8000 anomaly-detection

    curl http://localhost:8000/predict?category=bottle -F "file=@image.png"

### 6. Run Tests

    # Unit tests only (what CI runs)
    pytest tests/ -m "not integration"

    # Full suite, including data/model-dependent tests (requires dvc pull first)
    pytest tests/

### 7. (Optional) Retrain Models

    dvc repro train_patchcore
    dvc repro train_stfpm

MLflow experiment tracking is hosted on DagsHub. Set up a `.env` file
(see `.env.example`) with your DagsHub MLflow credentials to log runs.

### 8. (Optional) Generate Drift Monitoring Reports

Due to a dependency conflict between `evidently` and the main
project's numpy/pandas versions, monitoring runs in a separate,
isolated environment. See `src/monitoring/README.md` for full setup.

    bash run_monitoring_report.sh --category all

### Deployment Evidence

**Automated CI/CD pipeline (GitHub Actions):**
![CD Pipeline](docs\images\pipeline_actions.png)

**Live Cloud Run revision:**
![Cloud Run Console](docs\images\GCP.png)

**Live API responses:**
![Info Endpoint](docs\images\UI.png)
![Predict Endpoint](docs\images\prediction.png)

> Note: The Cloud Run service is not kept continuously running to
> avoid incurring idle costs. These screenshots were captured during
> active testing of the deployment pipeline. The full CI/CD pipeline
> can redeploy the service on demand in under 5 minutes.