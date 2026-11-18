FROM python:3.10-slim

WORKDIR /app

# System dependencies required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install only serving dependencies (not training/dev tools)
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Copy only what's needed for serving
COPY api/ api/
COPY src/inference/ src/inference/
COPY src/monitoring/ src/monitoring/
COPY src/__init__.py src/__init__.py

# NOTE: results/ (model checkpoints) is copied for local/demo purposes.
# In a production setup, this would be fetched at runtime from DVC/DagsHub
# or a model registry, not baked into the image.
RUN mkdir -p results
COPY results/ results/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]