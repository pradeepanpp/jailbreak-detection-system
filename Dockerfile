# Dockerfile
# Multi-stage build — keeps final image lean
# Stage 1: build dependencies
# Stage 2: runtime image

# ─────────────────────────────────────────────
# STAGE 1 — Builder
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (layer cache)
COPY requirements.txt .
# CPU build of PyTorch (~200 MB) instead of the default CUDA build (~2 GB)
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


# ─────────────────────────────────────────────
# STAGE 2 — Runtime
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/           src/
COPY configs/       configs/

# Trained models are NOT baked into the image (they are gitignored, so a
# CI build would silently ship without them). docker-compose mounts
# ./models read-only; /health returns 503 if a layer failed to load.
RUN mkdir -p models data/processed data/results logs && \
    chown -R appuser:appuser /app

# Environment
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PORT=8000 \
    HOST=0.0.0.0

USER appuser

EXPOSE 8000

# Health check — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["uvicorn", "src.jailbreak_detection.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "warning"]