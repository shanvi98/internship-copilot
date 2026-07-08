"""
FastAPI layer — exposes the agent pipeline as an HTTP API.

Endpoints:
  GET  /health              - liveness check
  POST /analyze             - run the full agent pipeline on a resume + JD
  POST /analyze/quick       - offline-only fast path (skills + gap, no LLM/embeddings)

Run locally with:
  uvicorn main:app --reload --port 8000
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.orchestrator import run_pipeline
from tools.jd_parser import extract_skills_offline
from tools.gap_analyzer import analyze_gap, prioritize_gaps

app = FastAPI(
    title="Internship Application Copilot API",
    description="Agentic pipeline that parses a JD, matches it against a resume, finds skill gaps, and rewrites bullets.",
    version="0.1.0",
)

# Allow the React dev server (and later, the deployed frontend) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # single-origin deployment (frontend served by this same app) makes this safe to leave open
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic client is optional — only constructed if a key is present in the
# environment. If absent, the pipeline just uses Ollama (the default backend).
_anthropic_client = None
if os.environ.get("ANTHROPIC_API_KEY"):
    import anthropic
    _anthropic_client = anthropic.Anthropic()


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=20, description="Full resume text, plain text.")
    jd_text: str = Field(..., min_length=20, description="Full job description text, plain text.")
    llm_backend: str = Field(default="ollama", description="'ollama' (free, local) or 'anthropic' (paid API).")


class AnalyzeResponse(BaseModel):
    steps: list[dict]
    jd_requirements: dict | None
    gap_analysis: dict | None
    prioritized_gaps: list[dict]
    rewritten_bullets: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if req.llm_backend not in ("ollama", "anthropic"):
        raise HTTPException(status_code=400, detail="llm_backend must be 'ollama' or 'anthropic'")
    if req.llm_backend == "anthropic" and _anthropic_client is None:
        raise HTTPException(
            status_code=400,
            detail="llm_backend='anthropic' requested but ANTHROPIC_API_KEY is not set on the server.",
        )

    state = run_pipeline(
        resume_text=req.resume_text,
        jd_text=req.jd_text,
        llm_backend=req.llm_backend,
        anthropic_client=_anthropic_client,
    )
    return state.summary()


@app.post("/analyze/quick")
def analyze_quick(req: AnalyzeRequest):
    """
    Fast, dependency-free path: skill extraction + gap analysis only.
    No Ollama, no embedding model, no LLM calls — always works, instant response.
    Useful as a fallback in the frontend if the full /analyze call is slow or Ollama is down.
    """
    gap = analyze_gap(req.resume_text, req.jd_text)
    return {
        "matched_skills": gap.matched_skills,
        "missing_skills": gap.missing_skills,
        "bonus_skills": gap.bonus_skills,
        "coverage_pct": gap.coverage_pct,
        "prioritized_gaps": prioritize_gaps(gap),
    }


# --- Serve the frontend ---
# Mounted last and at /ui (not "/") so it never shadows the API routes above.
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")
