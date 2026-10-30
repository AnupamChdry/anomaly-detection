# Anomaly Detection MLOps Pipeline

End-to-end MLOps pipeline for visual anomaly detection on industrial images
(MVTec AD dataset), covering data versioning, experiment tracking, CI/CD,
cloud deployment, and drift monitoring.

## Status: Week 2 Complete ✅ — Data Versioning Pipeline

## Tech Stack
- Data versioning: DVC (remote: DagsHub)
- Experiment tracking: MLflow (upcoming)
- Model: CNN / Autoencoder (PyTorch)
- API: FastAPI
- Containerization: Docker
- CI/CD: GitHub Actions
- Cloud: GCP (upcoming)
- Monitoring: Evidently AI (upcoming)

## Dataset
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) —
using 3 categories: `bottle`, `screw`, `cable`.

## Project Structure
\`\`\`
├── data/                   # DVC-tracked (raw + processed), not in Git directly
├── src/
│   ├── preprocessing/      # OpenCV-based image preprocessing
│   ├── training/           # Model training (upcoming - Week 3)
│   ├── inference/          # Inference logic (upcoming)
│   └── monitoring/         # Drift detection (upcoming - Week 7)
├── api/                    # FastAPI serving app
├── tests/                  # Unit tests
├── dvc.yaml                # DVC pipeline definition
├── dvc.lock                # DVC pipeline state (auto-generated)
└── .github/workflows/      # CI: lint, test, docker-build
\`\`\`

## Setup

### 1. Clone and install dependencies
\`\`\`bash
git clone https://github.com/AnupamChdry/anomaly-detection-mlops.git
cd anomaly-detection-mlops
python -m venv venv
source venv/Scripts/activate   # or venv/bin/activate on Mac/Linux
pip install -r requirements.txt
\`\`\`

### 2. Configure DVC remote (DagsHub)
\`\`\`bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-dagshub-username>
dvc remote modify origin --local password <your-dagshub-token>
\`\`\`

### 3. Pull versioned data
\`\`\`bash
dvc pull
\`\`\`

### 4. Reproduce the preprocessing pipeline
\`\`\`bash
dvc repro
\`\`\`

## Data Pipeline

\`\`\`mermaid
flowchart LR
    A[data/raw - MVTec AD] --> B[preprocess.py
    resize + normalize]
    B --> C[data/processed]
\`\`\`

Defined in \`dvc.yaml\`, tracked via DVC, stored remotely on DagsHub.

## Run API locally
\`\`\`bash
uvicorn api.main:app --reload
\`\`\`
Visit http://localhost:8000/docs for interactive API documentation.

## Run with Docker
\`\`\`bash
docker build -t anomaly-detection-app .
docker run -p 8000:8000 anomaly-detection-app
\`\`\`

## Testing
\`\`\`bash
pytest tests/ -v
\`\`\`

## CI/CD
This repo uses GitHub Actions for:
- Linting (flake8, black)
- Automated testing (pytest)
- Docker build verification