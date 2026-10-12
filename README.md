# Internship Application Copilot Agent

An agentic pipeline that parses a job description, matches it against your resume
using semantic embeddings, identifies skill gaps, and rewrites resume bullets to
better align with ATS keyword matching — while never inventing claims you didn't make.

## Features

- **Skill gap analysis** across 110+ skills with aliases (`k8s` → kubernetes, `sklearn` →
  scikit-learn, `LLMs`, `YOLOv8`, `RESTful APIs`, `Postgres`…), grouped by category, and
  context-aware so "R&D", "go to market" or "excel at" never count as skills.
- **Prioritized gaps** — critical (LLM must-have), high (listed under the JD's requirements),
  moderate, low (nice-to-have / "a plus"). Section detection works offline, no LLM needed.
- **ATS keyword coverage** — the JD's most prominent terms beyond tech skills (domain words,
  methods, artifacts) and which ones your resume is missing; benefits/EEO boilerplate ignored.
- **Resume checks** — weak openers ("Responsible for…"), unquantified bullets, first-person
  pronouns, overlong bullets, missing contact info or standard sections.
- **Match score (0–100)** blending weighted skill coverage, keyword coverage and semantic fit,
  so you can rank postings against each other.
- **Truthful bullet rewrites** — the LLM output is checked deterministically: a rewrite that
  adds a number or a tool your original bullet didn't mention is discarded, not shown.
- **Upload a PDF/DOCX resume**, or **fetch a JD from a posting URL** (uses the page's
  schema.org `JobPosting` data when present — Greenhouse, Lever, Ashby, Workday…).
- **Application tracker** (SQLite) — save analyses, move postings through
  saved → applied → interviewing → offer, see your response rate and the skills missing
  most often across everything you've applied to.
- **CLI** for scripted use, Markdown report export, and a no-build single-file dashboard.

## Why this is an agent, not a prompt chain

Each step in the pipeline retries on transient failure and **degrades gracefully**
instead of crashing: if the local LLM is down, JD extraction falls back to offline
keyword matching and the agent stops calling the LLM for the rest of the run (no
wasted retries and timeouts); if the embedding model can't load, bullets are still
picked and gap analysis still runs (it's pure logic); if a bullet rewrite fails — or
invents a claim — the original bullet is kept rather than dropped. Every step logs its
own status (`ok` / `degraded` / `skipped` / `failed`) and duration, so the final output
is honest about what actually ran — see `backend/agent/orchestrator.py`.

## Architecture

```
frontend/index.html          Single-file dashboard (no build step): agent trace rail, results, tracker
backend/
  tools/
    skill_taxonomy.py         Canonical skills, categories, aliases, context rules
    jd_parser.py              Offline skill/section/seniority extraction + LLM structured extraction
    resume_matcher.py         Sentence-embedding similarity (lazy-loaded) + skill evidence
    gap_analyzer.py           Skill diff: matched / missing / bonus + prioritization, pure logic
    keyword_analyzer.py       ATS keyword coverage, pure logic
    resume_linter.py          Resume quality checks, pure logic
    scoring.py                Blended 0-100 match score
    bullet_writer.py          LLM bullet rewriting + truthfulness guard
    llm.py                    Shared Ollama / Anthropic client: structured JSON, error taxonomy
    document_loader.py        PDF / DOCX / TXT text extraction
    jd_fetcher.py             Job posting URL fetcher (JSON-LD aware, SSRF-guarded)
  agent/
    orchestrator.py           Ties the tools together with retry, circuit breaker, degradation
    state.py                  Shared pipeline state + step logging/timing
  storage/tracker.py          SQLite application tracker
  config.py                   Environment-variable settings
  main.py                     FastAPI app; also serves the frontend at /ui
  cli.py                      Command-line interface
Dockerfile                    Single-image deploy (backend + frontend)
render.yaml                   One-click Render blueprint
```

**Pluggable LLM backend:** every LLM-dependent tool supports `backend="ollama"`
(free, local, default) or `backend="anthropic"` (paid API, optional). This was a
deliberate design choice to keep the whole pipeline runnable at zero cost. The
Anthropic path uses structured outputs (schema-guaranteed JSON), handles refusals,
and opts into server-side fallbacks.

## Running locally

```bash
# 1. Ollama (free local LLM) — optional; everything else works without it
ollama serve                      # separate terminal, keep running
ollama pull llama3.1:8b           # one-time, ~4.7GB

# 2. Backend
cd backend
pip install -r requirements.txt   # or requirements-core.txt to skip torch/embeddings
uvicorn main:app --reload --port 8000

# 3. Open the dashboard
open http://localhost:8000/ui/    # or double-click frontend/index.html
```

Visit `http://localhost:8000/docs` for the interactive API explorer.

### Command line

```bash
cd backend
python cli.py --resume resume.pdf --jd jd.txt                 # full agent, Markdown report
python cli.py --resume resume.pdf --jd-url https://...  --quick  # offline only, instant
python cli.py --resume resume.docx --jd jd.txt --json > report.json
python cli.py --resume resume.pdf --jd jd.txt --save --company Acme --status applied
```

### Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

The suite needs no LLM, network, or embedding model (all mocked), so it runs in a
couple of seconds — CI installs `requirements-dev.txt` and skips the multi-GB torch download.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model |
| `ANTHROPIC_API_KEY` | — | Enables the `anthropic` backend |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Claude model |
| `ANTHROPIC_EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` / `max` |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request LLM timeout |
| `COPILOT_DB_PATH` | `backend/copilot.db` | Application tracker database |
| `ALLOW_URL_FETCH` | `1` | Set `0` to disable `/jd/fetch` (recommended on public deployments) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## Deploying (free)

This repo deploys to [Render](https://render.com)'s free tier via the included
`render.yaml` blueprint — no credit card required. Connect the GitHub repo in the
Render dashboard and it builds from the Dockerfile automatically.

**Important limitation, stated honestly:** free hosting tiers don't have the
RAM/CPU to run Ollama. The deployed instance serves `/analyze/quick` (offline
skill-gap, keyword and resume analysis — instant, no LLM) reliably. The full agent
pipeline with live LLM bullet rewriting is meant to be demoed locally, where Ollama
runs on your own machine (or set `ANTHROPIC_API_KEY` on the host). This is a genuine
architecture trade-off, not a bug — and a good thing to be able to explain in an interview.

Free-tier services also sleep after 15 minutes of inactivity; the first request
after that takes 30-60 seconds to wake back up. Their disk is ephemeral too, so the
application tracker resets on redeploy, and it has no authentication — it's a
single-user tool, so keep a public deployment to the analysis on endpoints.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /capabilities` | Which LLM backends are usable right now (Ollama reachable / model pulled, Claude configured) |
| `POST /analyze` | Full agent pipeline (JD extraction, matching, gap analysis, keywords, resume checks, score, bullet rewrites) |
| `POST /analyze/quick` | Offline-only: skills, prioritized gaps, keywords, resume checks, score — no LLM/embeddings needed |
| `POST /rewrite` | Rewrite one bullet against a JD or skill list |
| `POST /extract-text` | Upload a PDF / DOCX / TXT / MD file, get plain text |
| `POST /jd/fetch` | Fetch a job description from a posting URL |
| `GET/POST /applications` | List (filter `?status=`, search `?q=`) / create tracked applications |
| `GET/PATCH/DELETE /applications/{id}` | Read (with saved analysis) / update / delete one |
| `GET /applications/stats` | Counts by status, average match score, response rate, most-missed skills |

## Roadmap

- [x] SQLite layer to track applications over time
- [x] JD scraping from a URL, not just pasted text
- [ ] React frontend (current dashboard is vanilla HTML/JS by design — see commit history for reasoning)
- [ ] Cover-letter drafting from the same truthfulness-guarded rewrite pipeline
