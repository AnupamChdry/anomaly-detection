## Week 6 — Cloud Deployment & CI/CD Automation (GCP Cloud Run)

### Overview

Week 6 extended the CI pipeline built in Week 5 (Docker image build + push to GitHub Container Registry) into a full **continuous deployment (CD)** pipeline: every merge to `main` automatically builds, scans, and deploys the latest model-serving container to **Google Cloud Run**, with zero manual cloud commands required.

The deployment uses **Workload Identity Federation (WIF)** for authentication — meaning no long-lived GCP service account key is ever stored as a GitHub secret. This was a deliberate security choice, informed directly by an earlier incident in this project (Week 2) where a DagsHub access token was accidentally committed and had to be rotated.

---

### Architecture

```mermaid
flowchart LR
    A[Developer merges PR<br/>develop → main] --> B[GitHub Actions triggered]
    B --> C[DVC pull model checkpoints<br/>from DagsHub remote]
    C --> D[Docker build<br/>CPU-only PyTorch image]
    D --> E[Push image to GHCR]
    E --> F[Trivy vulnerability scan]
    F --> G[Authenticate to GCP<br/>via Workload Identity Federation]
    G --> H[gcloud run deploy]
    H --> I[New revision live on Cloud Run]