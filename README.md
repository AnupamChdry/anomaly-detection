# Anomaly Detection MLOps Pipeline

End-to-end MLOps pipeline for visual anomaly detection on industrial images
(MVTec AD dataset), covering data versioning, experiment tracking, CI/CD,
cloud deployment, and drift monitoring.

## Status: Week 4 Complete ✅ — Packaging & Serving

## Tech Stack
- Data versioning: DVC (remote: DagsHub)
- Experiment tracking: MLflow (hosted via DagsHub)
- Models: PatchCore & STFPM (via Anomalib, PyTorch backend)
- API: FastAPI with model caching
- Containerization: Docker (optimized, CPU-only inference deps)
- CI/CD: GitHub Actions (with layer caching)
- Cloud: GCP (upcoming)
- Monitoring: Evidently AI (upcoming)

## Dataset
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) —
using 3 categories: `bottle`, `screw`, `cable`.

## Model Comparison

| Model | Category | Image AUROC | Pixel AUROC |
|-------|----------|-------------|-------------|
| PatchCore | bottle | 1.000 | 0.992 |
| PatchCore | screw  | 0.960 | 0.990 |
| PatchCore | cable  | 0.985 | 0.985 |
| STFPM     | bottle | 0.950 | 0.931 |
| STFPM     | screw  | 0.855 | 0.969 |
| STFPM     | cable  | 0.821 | 0.931 |

**PatchCore was selected for serving** due to consistently stronger performance.

## Serving Architecture

\`\`\`mermaid
flowchart LR
    A[Client] -->|POST /predict + image| B[FastAPI]
    B --> C[Load cached PatchCore model]
    C --> D[Preprocess image]
    D --> E[Run inference]
    E --> F[Return pred_score + pred_label]
\`\`\`

### Dependency Separation
To keep the serving container lightweight, dependencies are split:
- `requirements-dev.txt` — full stack (training, DVC, MLflow, lint/test tools)
- `requirements-serving.txt` — minimal inference-only deps (CPU-only PyTorch, FastAPI, Anomalib)

### Docker Build Optimization
| Build scenario | Time |
|---|---|
| Original (mixed deps, broken .dockerignore) | 25-30+ min |
| Fixed context + CPU-only torch (first run) | ~11 min |
| Cached rebuild (no changes) | ~3 sec |

Key fixes:
1. Corrected `.dockerignore` (was accidentally duplicated with `.dvcignore` content) — reduced build context from 2.62GB to 233KB
2. Forced CPU-only PyTorch build for serving (avoided multi-GB unnecessary CUDA library downloads)
3. GitHub Actions layer caching (`cache-from`/`cache-to: type=gha`)

## Project Structure
\`\`\`
├── data/                        # DVC-tracked raw dataset
├── src/
│   ├── preprocessing/           # OpenCV preprocessing pipeline
│   ├── training/                # PatchCore & STFPM training + MLflow tracking
│   ├── inference/               # predict.py - standalone, GPU/CPU-safe inference
│   └── monitoring/               # Drift detection (upcoming - Week 7)
├── api/main.py                  # FastAPI serving app with /predict endpoint
├── tests/                       # Unit + integration tests (incl. containerized API tests)
├── results/                     # DVC-tracked model checkpoints + evaluation outputs
├── dvc.yaml                     # Full pipeline: preprocess + train_patchcore + train_stfpm
├── Dockerfile                   # Lightweight serving image
├── .dockerignore / .dvcignore    # Separate, correctly-scoped ignore rules
└── .github/workflows/           # CI: lint, test, docker-build (cached)
\`\`\`

## Setup

### 1. Clone and install
\`\`\`bash
git clone https://github.com/AnupamChdry/anomaly-detection-mlops.git
cd anomaly-detection-mlops
python -m venv venv
source venv/Scripts/activate
pip install -r requirements-dev.txt
\`\`\`

### 2. Configure DVC + MLflow (DagsHub)
\`\`\`bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <token>
\`\`\`
\`\`\`
# .env
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<rep▍