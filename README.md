# Internship Application Copilot Agent

An agentic pipeline that parses a job description, matches it against your resume
using semantic embeddings, identifies skill gaps, and rewrites resume bullets to
better align with ATS keyword matching — while never inventing claims you didn't make.

## Why this is an agent, not a prompt chain

Each step in the pipeline retries on transient failure and **degrades gracefully**
instead of crashing: if the local LLM is down, JD extraction falls back to offline
keyword matching; if the embedding model can't load, gap analysis still runs (it's
pure logic); if a bullet rewrite fails, the original bullet is kept rather than
dropped. Every step logs its own status (`ok` / `degraded` / `failed`), so the final
output is honest about what actually ran — see `backend/agent/orchestrator.py`.

## Architecture

```
frontend/index.html        Single-file dashboard (no build step), agent trace rail
backend/
  tools/
    jd_parser.py            Offline skill extraction + LLM structured extraction
    resume_matcher.py       Sentence-embedding similarity (resume bullets vs JD)
    gap_analyzer.py         Skill diff: matched / missing / bonus, pure logic
    bullet_writer.py        LLM bullet rewriting, constrained to not invent claims
  agent/
    orchestrator.py         Ties the 4 tools together with retry + degradation
    state.py                Shared pipeline state + step logging
  main.py                   FastAPI app; also serves the frontend at /ui
Dockerfile                  Single-image deploy (backend + frontend)
render.yaml                 One-click Render blueprint
```

**Pluggable LLM backend:** every LLM-dependent tool supports `backend="ollama"`
(free, local, default) or `backend="anthropic"` (paid API, optional). This was a
deliberate design choice to keep the whole pipeline runnable at zero cost.

## Running locally

```bash
# 1. Ollama (free local LLM)
ollama serve                      # separate terminal, keep running
ollama pull llama3.1:8b           # one-time, ~4.7GB

# 2. Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. Open the dashboard
open ../frontend/index.html       # macOS; or just double-click it
```

Visit `http://localhost:8000/docs` for the interactive API explorer.

## Deploying (free)

This repo deploys to [Render](https://render.com)'s free tier via the included
`render.yaml` blueprint — no credit card required. Connect the GitHub repo in the
Render dashboard and it builds from the Dockerfile automatically.

**Important limitation, stated honestly:** free hosting tiers don't have the
RAM/CPU to run Ollama. The deployed instance serves `/analyze/quick` (offline
skill-gap analysis — instant, no LLM) reliably. The full agent pipeline with live
LLM bullet rewriting is meant to be demoed locally, where Ollama runs on your own
machine. This is a genuine architecture trade-off, not a bug — and a good thing to
be able to explain in an interview.

Free-tier services also sleep after 15 minutes of inactivity; the first request
after that takes 30-60 seconds to wake back up.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `POST /analyze` | Full agent pipeline (JD extraction, matching, gap analysis, bullet rewrites) |
| `POST /analyze/quick` | Offline-only: skill extraction + gap analysis, no LLM/embeddings needed |

## Roadmap

- [ ] SQLite layer to track applications over time (planned)
- [ ] JD scraping from a URL, not just pasted text
- [ ] React frontend (current dashboard is vanilla HTML/JS by design — see commit history for reasoning)
