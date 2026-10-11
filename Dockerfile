# Single-image deployment: FastAPI backend + statically-served frontend.
# Note: this image does NOT bundle Ollama. Free hosting tiers (Render, Railway
# free plan, etc.) don't have the RAM/CPU to run a local LLM anyway. Deployed,
# this serves the offline-only endpoints (/analyze/quick) reliably — plus the
# Claude backend if ANTHROPIC_API_KEY is set. The full agent pipeline with
# Ollama is meant to be demoed locally (README explains why — this is a
# legitimate, explainable architecture decision, not a limitation to hide).

FROM python:3.12-slim

WORKDIR /app

# System deps some ML wheels need to build/import cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user. The Hugging Face cache and the tracker's SQLite
# file both live in directories it owns.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/hf-cache \
    && chown -R app:app /app/data /app/hf-cache
ENV HF_HOME=/app/hf-cache \
    COPILOT_DB_PATH=/app/data/copilot.db \
    PYTHONUNBUFFERED=1

COPY backend/requirements.txt backend/requirements-core.txt ./

# Force the small CPU-only PyTorch wheel. Without this, pip pulls the default
# CUDA-enabled torch build (2GB+), which regularly exceeds free-tier build
# memory/disk limits — this is the most common cause of a generic
# "exited with status 1" build failure for sentence-transformers on Render/Railway free tiers.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time, not at first request —
# avoids a slow, timeout-risking cold start on a free-tier instance.
USER app
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=app:app backend/ ./backend/
COPY --chown=app:app frontend/ ./frontend/

WORKDIR /app/backend

# The tracker database lives in /app/data — mount a volume there to keep it across redeploys.
EXPOSE 8000

# Render (and most PaaS hosts) inject $PORT at runtime — fall back to 8000 for local docker run.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
