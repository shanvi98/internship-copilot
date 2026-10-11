"""
Command-line interface — run the copilot without the web dashboard.

Examples (from the backend/ directory):
  python cli.py --resume resume.pdf --jd jd.txt
  python cli.py --resume resume.docx --jd-url https://boards.greenhouse.io/acme/jobs/123
  python cli.py --resume resume.txt --jd jd.txt --quick            # offline only, instant
  python cli.py --resume resume.txt --jd jd.txt --backend anthropic --json > report.json
  python cli.py --resume resume.txt --jd jd.txt --save --company Acme --status applied
"""

import argparse
import json
import os
import sys
from pathlib import Path

from storage.tracker import STATUSES
from tools.document_loader import DocumentError, extract_text


def _read(path: str) -> str:
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: file not found: {path}")
    try:
        return extract_text(p.name, p.read_bytes())
    except DocumentError as e:
        sys.exit(f"error: {path}: {e}")


def to_markdown(result: dict) -> str:
    """Human-readable report for an /analyze or /analyze/quick style result."""
    lines = []
    overview = result.get("jd_overview") or {}
    title = overview.get("role_title") or "Job description"
    lines.append(f"# Match report — {title}")
    if overview.get("company"):
        lines.append(f"_{overview['company']}_")

    score = result.get("match_score") or {}
    if score:
        comps = ", ".join(f"{k} {v:.0f}" for k, v in score.get("components", {}).items())
        lines.append(f"\n**Match score: {score.get('score')}/100 ({score.get('label')})** — {comps}")

    gap = result.get("gap_analysis") or {}
    if gap:
        lines.append(f"\n## Skills ({gap.get('coverage_pct', 0)}% of JD skills covered)")
        lines.append(f"- **Matched:** {', '.join(gap.get('matched_skills', [])) or 'none'}")
        lines.append(f"- **Missing:** {', '.join(gap.get('missing_skills', [])) or 'none'}")
        lines.append(f"- **Bonus:** {', '.join(gap.get('bonus_skills', [])) or 'none'}")

    gaps = result.get("prioritized_gaps") or []
    if gaps:
        lines.append("\n## Gaps to address (highest priority first)")
        for g in gaps:
            lines.append(f"- [{g['priority']}] **{g['skill']}** — {g.get('reason', '')}")

    kw = result.get("keyword_analysis") or {}
    if kw.get("missing_keywords"):
        lines.append(f"\n## ATS keywords ({kw.get('keyword_match_pct', 0)}% present)")
        lines.append("Missing from your resume: " + ", ".join(kw["missing_keywords"]))

    checks = (result.get("resume_checks") or {}).get("issues") or []
    if checks:
        lines.append("\n## Resume checks")
        for issue in checks:
            where = f" — “{issue['text'][:80]}”" if issue.get("text") else ""
            lines.append(f"- {issue['suggestion']}{where}")

    bullets = [b for b in result.get("rewritten_bullets") or [] if b.get("status") == "ok"]
    if bullets:
        lines.append("\n## Suggested bullet rewrites")
        for b in bullets:
            lines.append(f"- ~~{b['original']}~~\n  → {b['rewritten_bullet']}")

    steps = result.get("steps") or []
    if steps:
        lines.append("\n## Agent trace")
        for s in steps:
            detail = f" — {s['detail']}" if s.get("detail") else ""
            lines.append(f"- `{s['step']}` [{s['status']}] {s.get('duration_ms', 0)}ms{detail}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internship Application Copilot — analyze a resume against a job description.")
    parser.add_argument("--resume", required=True, help="Resume file (.pdf, .docx, .txt, .md)")
    jd_group = parser.add_mutually_exclusive_group(required=True)
    jd_group.add_argument("--jd", help="Job description file (.pdf, .docx, .txt, .md)")
    jd_group.add_argument("--jd-url", help="Job posting URL to fetch the description from")
    parser.add_argument("--quick", action="store_true", help="Offline analysis only (no LLM, no embeddings)")
    parser.add_argument("--backend", choices=["ollama", "anthropic"], default="ollama", help="LLM backend for the full pipeline")
    parser.add_argument("--rewrites", type=int, default=3, help="How many bullets to rewrite (0 disables)")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip the sentence-embedding model")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a Markdown report")
    parser.add_argument("--save", action="store_true", help="Save the result to the application tracker")
    parser.add_argument("--company", default="", help="Company name (with --save)")
    parser.add_argument("--role", default="", help="Role title (with --save; defaults to the detected title)")
    parser.add_argument("--status", default="saved", choices=STATUSES, help="Tracker status (with --save)")
    args = parser.parse_args(argv)

    resume_text = _read(args.resume)
    jd_url = ""
    if args.jd_url:
        from tools.jd_fetcher import FetchError, fetch_job_description

        try:
            fetched = fetch_job_description(args.jd_url)
        except FetchError as e:
            sys.exit(f"error: {e}")
        jd_text, jd_url = fetched["text"], fetched["url"]
        args.company = args.company or fetched.get("company", "")
    else:
        jd_text = _read(args.jd)

    if args.quick:
        from agent.orchestrator import run_offline_analysis

        result = run_offline_analysis(resume_text, jd_text)
    else:
        from agent.orchestrator import run_pipeline
        from tools.llm import make_anthropic_client

        client = None
        if args.backend == "anthropic":
            client = make_anthropic_client()
            if client is None or not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                sys.exit("error: --backend anthropic needs ANTHROPIC_API_KEY set")
        result = run_pipeline(
            resume_text, jd_text, llm_backend=args.backend, anthropic_client=client,
            max_rewrites=args.rewrites, use_embeddings=not args.no_embeddings,
        ).summary()

    if args.save:
        from config import settings
        from storage.tracker import Tracker

        gap = result.get("gap_analysis") or {}
        item = Tracker(settings.db_path).create(
            company=args.company or (result.get("jd_overview") or {}).get("company", ""),
            role=args.role or (result.get("jd_overview") or {}).get("role_title") or "",
            url=jd_url,
            status=args.status,
            match_score=(result.get("match_score") or {}).get("score"),
            coverage_pct=gap.get("coverage_pct"),
            missing_skills=gap.get("missing_skills", []),
            analysis=result,
        )
        print(f"Saved to tracker as #{item['id']}", file=sys.stderr)

    print(json.dumps(result, indent=2) if args.json else to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
