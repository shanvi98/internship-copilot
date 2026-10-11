"""
Agent Orchestrator
This is the "agent" part: it plans a fixed sequence of tool calls, but unlike a
plain prompt chain, each step:
  - retries on transient failure (e.g. a malformed LLM response, a timeout)
  - stops calling the LLM altogether once it's clearly unavailable (Ollama not
    running, no API key) instead of paying retries + timeouts on every step
  - degrades gracefully to a cheaper/offline fallback instead of crashing the
    whole pipeline (e.g. LLM extraction fails -> fall back to offline skill match)
  - logs what actually happened, so the final output is honest about which
    steps used the LLM vs. the offline fallback

This graceful-degradation behavior is the thing to point at in an interview when
asked "what makes this an agent and not just a script."
"""

import time
from concurrent.futures import ThreadPoolExecutor

from agent.state import AgentState
from tools.bullet_writer import UnsupportedClaimError, rewrite_bullet
from tools.gap_analyzer import analyze_gap, prioritize_gaps
from tools.jd_parser import extract_requirements_llm, extract_skills_offline
from tools.keyword_analyzer import analyze_keywords
from tools.llm import LLMRefusal, LLMUnavailable
from tools.resume_linter import lint_resume
from tools.resume_matcher import match_resume_to_jd, split_resume_into_bullets
from tools.scoring import compute_match_score

# Errors that won't go away by asking again.
_NON_RETRYABLE = (LLMUnavailable, LLMRefusal)


def _with_retry(fn, *args, retries: int = 2, delay_seconds: float = 1.5, **kwargs):
    """Run fn with a small number of retries. Raises the last exception if all attempts fail."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except _NON_RETRYABLE:
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay_seconds)
    raise last_exc


def run_offline_analysis(resume_text: str, jd_text: str) -> dict:
    """
    The instant, dependency-free subset of the pipeline: skill gap, ATS keywords,
    resume checks and a match score. Used by /analyze/quick and the CLI's --quick mode.
    """
    jd = extract_skills_offline(jd_text)
    gap = analyze_gap(resume_text, jd_text)
    keywords = analyze_keywords(resume_text, jd_text)
    gap_dict = gap.to_dict()
    return {
        "jd_overview": {"role_title": jd.role_title, "seniority": jd.seniority_hint},
        "gap_analysis": gap_dict,
        "prioritized_gaps": prioritize_gaps(gap, requirement_levels=gap.requirement_levels),
        "keyword_analysis": keywords,
        "resume_checks": lint_resume(resume_text),
        "match_score": compute_match_score(gap_dict, keyword_match_pct=keywords["keyword_match_pct"]),
    }


def _rewrite_one(bullet: str, skills: list[str], llm_backend: str, client) -> dict:
    try:
        rewritten = _with_retry(rewrite_bullet, bullet, skills, backend=llm_backend, client=client, retries=2)
        return {"original": bullet, **rewritten, "status": "ok"}
    except Exception as e:
        # degrade to "keep original" rather than dropping the bullet entirely
        if isinstance(e, UnsupportedClaimError):
            note = f"Rewrite discarded because it {e} — kept your original wording."
        else:
            note = f"Rewrite failed, kept original: {e}"
        return {
            "original": bullet,
            "rewritten_bullet": bullet,
            "keywords_incorporated": [],
            "confidence_note": note,
            "status": "unavailable" if isinstance(e, LLMUnavailable) else "failed",
        }


def run_pipeline(
    resume_text: str,
    jd_text: str,
    llm_backend: str = "ollama",
    anthropic_client=None,
    max_rewrites: int = 3,
    use_embeddings: bool = True,
) -> AgentState:
    state = AgentState(resume_text=resume_text, jd_text=jd_text, llm_backend=llm_backend)

    # --- Step 1: JD requirement extraction ---
    # Always run the offline pass first — it's free and instant, and becomes
    # our fallback if the LLM step fails.
    state.start_step()
    offline_result = extract_skills_offline(jd_text)
    state.jd_skills_offline = offline_result.matched_skills
    state.jd_overview = {"role_title": offline_result.role_title, "seniority": offline_result.seniority_hint}

    try:
        llm_result = _with_retry(
            extract_requirements_llm, jd_text, backend=llm_backend, client=anthropic_client, retries=2
        )
        state.jd_requirements = llm_result
        if llm_result.get("role_title"):
            state.jd_overview["role_title"] = llm_result["role_title"]
        if llm_result.get("company"):
            state.jd_overview["company"] = llm_result["company"]
        state.log("jd_extraction", "ok", f"LLM extraction succeeded via {llm_backend}")
    except Exception as e:
        if isinstance(e, LLMUnavailable):
            state.llm_unavailable_reason = str(e)
        state.jd_requirements = None
        state.log("jd_extraction", "degraded", f"LLM extraction failed ({e}); using offline skill match only")

    # --- Step 2: Resume-JD semantic matching ---
    state.start_step()
    if not use_embeddings:
        bullets = split_resume_into_bullets(resume_text)
        state.resume_match = {"jd_similarity_score": None, "top_bullets": bullets[:5], "skill_evidence": {}}
        state.log("resume_matching", "skipped", "Embeddings disabled; using the resume's first bullets")
    else:
        try:
            match = match_resume_to_jd(resume_text, jd_text, jd_skills=state.jd_skills_offline)
            state.resume_match = {
                "jd_similarity_score": match.jd_similarity_score,
                "top_bullets": match.top_bullets,
                "bullet_scores": [{"bullet": b, "score": s} for b, s in match.bullet_scores],
                "skill_evidence": match.skill_evidence,
            }
            state.log("resume_matching", "ok")
        except Exception as e:
            # Still give step 4 something to work with: the resume's own first bullets.
            bullets = split_resume_into_bullets(resume_text)
            state.resume_match = {"jd_similarity_score": None, "top_bullets": bullets[:5], "skill_evidence": {}} if bullets else None
            state.log("resume_matching", "degraded" if bullets else "failed", f"Semantic matching unavailable: {e}")

    # --- Step 3: Gap analysis, keywords, resume checks (pure logic, should basically never fail) ---
    state.start_step()
    try:
        reqs = state.jd_requirements or {}
        gap = analyze_gap(resume_text, jd_text, extra_jd_skills=reqs.get("must_have_skills"))
        state.gap_analysis = gap.to_dict()
        state.prioritized_gaps = prioritize_gaps(
            gap,
            must_have_skills=reqs.get("must_have_skills"),
            nice_to_have_skills=reqs.get("nice_to_have_skills"),
            requirement_levels=gap.requirement_levels,
        )
        state.keyword_analysis = analyze_keywords(resume_text, jd_text)
        state.resume_checks = lint_resume(resume_text)
        state.match_score = compute_match_score(
            state.gap_analysis,
            keyword_match_pct=state.keyword_analysis["keyword_match_pct"],
            semantic_similarity=(state.resume_match or {}).get("jd_similarity_score"),
        )
        state.log("gap_analysis", "ok")
    except Exception as e:
        state.log("gap_analysis", "failed", str(e))

    # --- Step 4: Bullet rewriting ---
    # Only rewrite the top bullets from Step 2 (the ones most relevant to this JD) —
    # no point spending LLM calls rewriting irrelevant bullets.
    state.start_step()
    top_bullets = (state.resume_match or {}).get("top_bullets", [])[:max_rewrites]
    if max_rewrites <= 0:
        state.log("bullet_rewriting", "skipped", "Bullet rewriting disabled for this request")
    elif not top_bullets:
        state.log("bullet_rewriting", "degraded", "No resume bullets available to rewrite")
    elif state.llm_unavailable_reason:
        state.log("bullet_rewriting", "skipped", f"LLM unavailable: {state.llm_unavailable_reason}")
    else:
        # Give the rewriter the JD's own vocabulary: LLM must-haves when we have them,
        # plus the offline taxonomy matches.
        jd_skill_list = list(dict.fromkeys(
            (state.jd_requirements or {}).get("must_have_skills", []) + state.jd_skills_offline
        ))
        with ThreadPoolExecutor(max_workers=min(3, len(top_bullets))) as pool:
            results = list(pool.map(lambda b: _rewrite_one(b, jd_skill_list, llm_backend, anthropic_client), top_bullets))
        state.rewritten_bullets = results

        succeeded = sum(1 for r in results if r["status"] == "ok")
        if succeeded == len(results):
            state.log("bullet_rewriting", "ok", f"Rewrote {succeeded} bullet(s)")
        elif succeeded:
            state.log("bullet_rewriting", "degraded", f"Rewrote {succeeded} of {len(results)} bullets; kept originals for the rest")
        else:
            state.log("bullet_rewriting", "failed", "No bullet rewrites succeeded; kept originals")

    return state


if __name__ == "__main__":
    sample_resume = """
    Built a HybridRAG customer complaint classifier using FAISS, Ollama, and Streamlit.
    Developed HoopSense AI, a full-stack basketball shooting analysis platform using React, FastAPI, MediaPipe, YOLOv8, and XGBoost.
    Built a resume parser and job matcher using NLP techniques and Scikit-learn.
    """
    sample_jd = """
    Machine Learning Engineering Intern - Summer 2027
    You'll work with PyTorch, build RAG pipelines using LangChain and FAISS,
    and deploy models via FastAPI and Docker on AWS.
    """

    result_state = run_pipeline(sample_resume, sample_jd, llm_backend="ollama")
    import json
    print(json.dumps(result_state.summary(), indent=2))
