# Single-image deployment: FastAPI backend + statically-served frontend.
# Note: this image does NOT bundle Ollama. Free hosting tiers (Render, Railway
# free plan, etc.) don't have the RAM/CPU to run a local LLM anyway. Deployed,
# this serves the offline-only endpoints (/analyze/quick) reliably. The full
# agent pipeline with Ollama is meant to be demoed locally (README explains why —
# this is a legitimate, explainable architecture decision, not a limitation to hide).

FROM python:3.12-slim

WORKDIR /app

# System deps some ML wheels need to build/import cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time, not at first request —
# avoids a slow, timeout-risking cold start on a free-tier instance.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

EXPOSE 8000

# Render (and most PaaS hosts) inject $PORT at runtime — fall back to 8000 for local docker run.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
