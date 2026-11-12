## Status: Week 5 Complete ✅ — CI/CD & Container Registry

## CI/CD Pipeline

On every push to `main`, GitHub Actions automatically:
1. Pulls trained model checkpoints from DagsHub (via DVC)
2. Builds the Docker image (using cached layers for speed)
3. Tags the image with both `latest` and `<branch>-<commit-sha>` for traceability
4. Pushes to GitHub Container Registry (GHCR)
5. Scans the published image for vulnerabilities (Trivy)

\`\`\`mermaid
flowchart LR
    A[Push to main] --> B[dvc pull checkpoints]
    B --> C[Build + tag image]
    C --> D[Push to GHCR]
    D --> E[Trivy vulnerability scan]
\`\`\`

## Pulling the Published Image

\`\`\`bash
docker pull ghcr.io/anupamchdry/anomaly-detection:latest
docker run -p 8000:8000 ghcr.io/anupamchdry/anomaly-detection:latest
\`\`\`

## Verified Reproducibility

The full CI/CD pipeline was verified end-to-end:
- Built image in CI (fresh environment, no local cache)
- Published to GHCR
- Pulled fresh (not from local build cache) on a separate machine/session
- Confirmed `/predict` returns correct results for both normal and anomalous images

This process caught and fixed a real bug: the checkpoint lookup logic
needed to check anomalib's `latest` folder (used by the CI pipeline)
in addition to versioned folders (`v0`, `v1`, used in local training).