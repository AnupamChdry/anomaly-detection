# Anomaly Detection MLOps Pipeline

End-to-end MLOps pipeline for visual anomaly detection on industrial images
(MVTec AD dataset), covering data versioning, experiment tracking, CI/CD,
cloud deployment, and drift monitoring.

## Status: Week 3 Complete ✅ — Model Training & Experiment Tracking

## Tech Stack
- Data versioning: DVC (remote: DagsHub)
- Experiment tracking: MLflow (hosted via DagsHub)
- Models: PatchCore & STFPM (via Anomalib, PyTorch backend)
- API: FastAPI
- Containerization: Docker (with GitHub Actions layer caching)
- CI/CD: GitHub Actions
- Cloud: GCP (upcoming)
- Monitoring: Evidently AI (upcoming)

## Dataset
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) —
using 3 categories: `bottle`, `screw`, `cable`.

## Model Comparison

Two state-of-the-art anomaly detection approaches were trained and evaluated
on all 3 categories, tracked via MLflow:

| Model | Category | Image AUROC | Pixel AUROC |
|-------|----------|-------------|-------------|
| PatchCore | bottle | 1.000 | 0.992 |
| PatchCore | screw  | 0.960 | 0.990 |
| PatchCore | cable  | 0.985 | 0.985 |
| STFPM     | bottle | 0.950 | 0.931 |
| STFPM     | screw  | 0.855 | 0.969 |
| STFPM     | cable  | 0.821 | 0.931 |

**Key finding**: PatchCore (memory-bank based) consistently outperformed
STFPM (student-teacher based) across all categories, particularly on
structurally complex categories like `cable` — consistent with published
benchmarks on MVTec AD.

## Project Structure
\`\`\`
├── data/                   # DVC-tracked (raw), not in git directly
├── src/
│   ├── preprocessing/      # OpenCV-based image preprocessing
│   ├── training/           # PatchCore & STFPM training/evaluation + MLflow
│   ├── inference/          # Inference logic (upcoming)
│   └── monitoring/         # Drift detection (upcoming - Week 7)
├── api/                    # FastAPI serving app
├── tests/                  # Unit + integration tests
├── results/                # DVC-tracked model outputs/summaries
├── dvc.yaml                # Full pipeline: preprocess + both training stages
├── dvc.lock                # DVC pipeline state (auto-generated)
└── .github/workflows/      # CI: lint, test, docker-build (with caching)
\`\`\`

## Setup

### 1. Clone and install dependencies
\`\`\`bash
git clone https://github.com/AnupamChdry/anomaly-detection-mlops.git
cd anomaly-detection-mlops
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
\`\`\`

### 2. Configure DVC remote (DagsHub)
\`\`\`bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-dagshub-username>
dvc remote modify origin --local password <your-dagshub-token>
\`\`\`

### 3. Configure MLflow tracking (.env file)
\`\`\`
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<token>
\`\`\`

### 4. Pull versioned data
\`\`\`bash
dvc pull
\`\`\`

### 5. Reproduce the full pipeline
\`\`\`bash
dvc repro
\`\`\`

## Pipeline Architecture

\`\`\`mermaid
flowchart LR
    A[data/raw - MVTec AD] --> B[preprocess]
    B --> C[data/processed]
    A --> D[train_patchcore]
    A --> E[train_stfpm]
    D --> F[MLflow: patchcore results]
    E --> G[MLflow: stfpm results]
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

## Testing
\`\`\`bash
pytest tests/ -v -m "not integrat▍