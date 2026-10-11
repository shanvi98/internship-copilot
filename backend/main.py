"""
FastAPI layer — exposes the agent pipeline as an HTTP API.

Endpoints:
  GET    /health                 - liveness check (cheap, used by the host's health probe)
  GET    /capabilities           - which LLM backends are usable right now
  POST   /analyze                - run the full agent pipeline on a resume + JD
  POST   /analyze/quick          - offline-only fast path (skills, gaps, keywords, resume checks — no LLM/embeddings)
  POST   /rewrite                - rewrite a single bullet against a JD
  POST   /extract-text           - upload a PDF/DOCX/TXT resume or JD, get plain text back
  POST   /jd/fetch               - fetch a job description from a posting URL
  GET    /applications           - list tracked applications (filter by ?status= or ?q=)
  POST   /applications           - track a new application (optionally with its analysis)
  GET    /applications/stats     - pipeline counts, avg match score, most-missed skills
  GET    /applications/{id}      - one application, including its saved analysis
  PATCH  /applications/{id}      - update status/notes/etc.
  DELETE /applications/{id}      - stop tracking an application

Run locally with:
  uvicorn main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agent.orchestrator import run_offline_analysis, run_pipeline
from config import settings
from storage.tracker import STATUSES, Tracker
from tools.bullet_writer import UnsupportedClaimError, rewrite_bullet
from tools.document_loader import DocumentError, MAX_UPLOAD_BYTES, extract_text
from tools.jd_fetcher import FetchError, fetch_job_description
from tools.jd_parser import extract_skills_offline
from tools.llm import LLMError, make_anthropic_client, ollama_status
from tools.resume_matcher import is_model_loaded

__version__ = "0.2.0"

# Anthropic client is optional — only constructed if credentials are present in
# the environment. If absent, the pipeline just uses Ollama (the default backend).
_anthropic_client = (
    make_anthropic_client()
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    else None
)

tracker: Tracker | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global tracker
    tracker = Tracker(settings.db_path)
    yield


app = FastAPI(
    title="Internship Application Copilot API",
    description="Agentic pipeline that parses a JD, matches it against a resume, finds skill gaps, and rewrites bullets.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

Backend = Literal["ollama", "anthropic"]
Status = Literal["saved", "applied", "interviewing", "offer", "rejected", "withdrawn"]


class QuickAnalyzeRequest(BaseModel):
    resume_text: str = Field(
        ..., min_length=20, max_length=15000,
        description="Full resume text, plain text. Capped at ~15000 chars (a dense 3-page resume) to bound embedding/LLM cost per request.",
    )
    jd_text: str = Field(
        ..., min_length=20, max_length=15000,
        description="Full job description text, plain text.",
    )

    @field_validator("resume_text", "jd_text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("must contain at least 20 non-whitespace characters")
        return v


class AnalyzeRequest(QuickAnalyzeRequest):
    llm_backend: Backend = Field(default="ollama", description="'ollama' (free, local) or 'anthropic' (paid API).")
    max_rewrites: int = Field(default=3, ge=0, le=8, description="How many top-matching bullets to rewrite (0 disables).")
    use_embeddings: bool = Field(default=True, description="Disable to skip the sentence-embedding model (faster on small machines).")


class AnalyzeResponse(BaseModel):
    steps: list[dict]
    llm_backend: str
    jd_overview: dict
    jd_requirements: dict | None
    resume_match: dict | None
    gap_analysis: dict | None
    prioritized_gaps: list[dict]
    keyword_analysis: dict | None
    resume_checks: dict | None
    match_score: dict | None
    rewritten_bullets: list[dict]


class RewriteRequest(BaseModel):
    bullet: str = Field(..., min_length=10, max_length=600)
    jd_text: str | None = Field(default=None, max_length=15000, description="JD to tailor against (skills are extracted from it).")
    jd_skills: list[str] | None = Field(default=None, description="Explicit skills to target, instead of / in addition to jd_text.")
    llm_backend: Backend = "ollama"


class FetchRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)


class ApplicationCreate(BaseModel):
    company: str = Field(default="", max_length=200)
    role: str = Field(default="", max_length=200)
    url: str = Field(default="", max_length=2000)
    status: Status = "saved"
    notes: str = Field(default="", max_length=5000)
    deadline: str | None = Field(default=None, description="ISO date, e.g. 2027-01-15")
    applied_on: str | None = None
    match_score: int | None = Field(default=None, ge=0, le=100)
    coverage_pct: float | None = Field(default=None, ge=0, le=100)
    missing_skills: list[str] = Field(default_factory=list)
    analysis: dict | None = Field(default=None, description="Optional full /analyze or /analyze/quick response to keep with the application.")


class ApplicationUpdate(BaseModel):
    company: str | None = Field(default=None, max_length=200)
    role: str | None = Field(default=None, max_length=200)
    url: str | None = Field(default=None, max_length=2000)
    status: Status | None = None
    notes: str | None = Field(default=None, max_length=5000)
    deadline: str | None = None
    applied_on: str | None = None
    match_score: int | None = Field(default=None, ge=0, le=100)


def _require_anthropic(backend: str):
    if backend == "anthropic" and _anthropic_client is None:
        raise HTTPException(
            status_code=400,
            detail="llm_backend='anthropic' requested but ANTHROPIC_API_KEY is not set on the server.",
        )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/capabilities")
def capabilities():
    """What this server can do right now — the dashboard uses it to pick a sensible default backend."""
    return {
        "version": __version__,
        "ollama": ollama_status(),
        "anthropic": {"configured": _anthropic_client is not None, "model": settings.anthropic_model},
        "embeddings": {"loaded": is_model_loaded()},
        "url_fetch": settings.allow_url_fetch,
        "statuses": list(STATUSES),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    _require_anthropic(req.llm_backend)
    state = run_pipeline(
        resume_text=req.resume_text,
        jd_text=req.jd_text,
        llm_backend=req.llm_backend,
        anthropic_client=_anthropic_client,
        max_rewrites=req.max_rewrites,
        use_embeddings=req.use_embeddings,
    )
    return state.summary()


@app.post("/analyze/quick")
def analyze_quick(req: QuickAnalyzeRequest):
    """
    Fast, dependency-free path: skill extraction, gap analysis, ATS keywords and resume checks.
    No Ollama, no embedding model, no LLM calls — always works, instant response.
    Useful as a fallback in the frontend if the full /analyze call is slow or Ollama is down.
    """
    result = run_offline_analysis(req.resume_text, req.jd_text)
    gap = result["gap_analysis"]
    # Top-level gap fields kept for backwards compatibility with v0.1 clients.
    return {
        "matched_skills": gap["matched_skills"],
        "missing_skills": gap["missing_skills"],
        "bonus_skills": gap["bonus_skills"],
        "coverage_pct": gap["coverage_pct"],
        **result,
    }


@app.post("/rewrite")
def rewrite(req: RewriteRequest):
    """Rewrite one bullet on demand — e.g. a bullet the full pipeline didn't pick."""
    _require_anthropic(req.llm_backend)
    skills = list(req.jd_skills or [])
    if req.jd_text:
        skills += extract_skills_offline(req.jd_text).matched_skills
    skills = list(dict.fromkeys(skills))
    if not skills:
        raise HTTPException(status_code=400, detail="Provide jd_text or jd_skills to tailor the bullet toward.")
    try:
        result = rewrite_bullet(req.bullet, skills, backend=req.llm_backend, client=_anthropic_client)
    except UnsupportedClaimError as e:
        raise HTTPException(status_code=422, detail=f"The rewrite {e}, so it was discarded. Try again or edit manually.") from e
    except (LLMError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"Rewrite unavailable: {e}") from e
    return {"original": req.bullet, **result}


@app.post("/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        text = extract_text(file.filename or "", data)
    except DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"filename": file.filename, "chars": len(text), "text": text}


@app.post("/jd/fetch")
def fetch_jd(req: FetchRequest):
    if not settings.allow_url_fetch:
        raise HTTPException(status_code=403, detail="Fetching URLs is disabled on this server. Paste the JD text instead.")
    try:
        return fetch_job_description(req.url)
    except FetchError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# --- Application tracker ---

def _tracker() -> Tracker:
    if tracker is None:  # only possible if the lifespan didn't run
        raise HTTPException(status_code=503, detail="Tracker not initialized")
    return tracker


@app.get("/applications")
def list_applications(status: Status | None = None, q: str | None = Query(default=None, max_length=200)):
    return _tracker().list(status=status, query=q)


@app.post("/applications", status_code=201)
def create_application(req: ApplicationCreate):
    return _tracker().create(**req.model_dump())


@app.get("/applications/stats")
def application_stats():
    return _tracker().stats()


@app.get("/applications/{app_id}")
def get_application(app_id: int):
    item = _tracker().get(app_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return item


@app.patch("/applications/{app_id}")
def update_application(app_id: int, req: ApplicationUpdate):
    item = _tracker().update(app_id, **req.model_dump(exclude_unset=True))
    if item is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return item


@app.delete("/applications/{app_id}", status_code=204)
def delete_application(app_id: int):
    if not _tracker().delete(app_id):
        raise HTTPException(status_code=404, detail="Application not found")


# --- Serve the frontend ---
# Mounted last and at /ui (not "/") so it never shadows the API routes above.
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")
