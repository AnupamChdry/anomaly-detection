# Anomaly Detection MLOps Pipeline

End-to-end MLOps pipeline for visual anomaly detection on industrial images
(MVTec AD dataset), covering data versioning, experiment tracking, CI/CD,
cloud deployment, and drift monitoring.

## Status: Week 1 Complete ✅ — Foundations (CI/CD basics)

## Tech Stack
- Data versioning: DVC (upcoming)
- Experiment tracking: MLflow (upcoming)
- Model: CNN / Autoencoder (PyTorch)
- API: FastAPI
- Containerization: Docker
- CI/CD: GitHub Actions
- Cloud: GCP (upcoming)
- Monitoring: Evidently AI (upcoming)

## Setup
\`\`\`bash
pip install -r requirements.txt
pytest tests/ -v
\`\`\`

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

## CI/CD
This repo uses GitHub Actions for:
- Linting (flake8, black)
- Automated testing (pytest)
- Docker build verification

## drive id: `1ZruS2WMccP0sOI1FAAt0SvAOnJlhDlHI`