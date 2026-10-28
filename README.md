# Anomaly Detection MLOps Pipeline

End-to-end MLOps pipeline for visual anomaly detection on industrial images
(MVTec AD dataset), covering data versioning, experiment tracking, CI/CD,
cloud deployment, and drift monitoring.

## Status: 🚧 In Progress

## Tech Stack
- Data versioning: DVC
- Experiment tracking: MLflow
- Model: CNN / Autoencoder (PyTorch)
- API: FastAPI
- Containerization: Docker
- CI/CD: GitHub Actions
- Cloud: GCP (Cloud Run)
- Monitoring: Evidently AI

## Setup
\`\`\`bash
pip install -r requirements.txt
pytest tests/
\`\`\`