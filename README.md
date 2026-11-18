## Week 7 — Model & Data Monitoring (Evidently AI)

### Overview

With the model deployed and serving predictions (Week 6), Week 7 adds
**drift monitoring** — tracking whether incoming production data and
model behavior remain statistically consistent with the training
distribution over time. This is a critical capability for maintaining
trust in a deployed model: a model doesn't become "wrong" on its own,
but the world around it can shift (lighting conditions, camera
hardware, product variations) in ways that silently degrade prediction
reliability.

### Architecture

```mermaid
flowchart TD
    A["/predict API call"] --> B["Run inference<br/>(PatchCore)"]
    B --> C["Log prediction to<br/>predictions_log.csv"]
    C --> D["Return response<br/>(unchanged)"]

    E["Training data<br/>(data/raw/category/train/good)"] --> F["build_reference.py<br/>(run once per model version)"]
    F --> G["reference_category.csv"]

    G --> H["generate_report.py<br/>(on-demand, isolated venv)"]
    C --> H
    H --> I["Timestamped Evidently<br/>HTML drift report"]