"""
Agent Orchestrator
This is the "agent" part: it plans a fixed sequence of tool calls, but unlike a
plain prompt chain, each step:
  - retries on transient failure (e.g. Ollama cold start, network hiccup)
  - degrades gracefully to a cheaper/offline fallback instead of crashing the
    whole pipeline (e.g. LLM extraction fails -> fall back to offline skill match)
  - logs what actually happened, so the final output is honest about which
    steps used the LLM vs. the offline fallback

This graceful-degradation behavior is the thing to point at in an interview when
asked "what makes this an agent and not just a script."
"""

import time
from agent.state import AgentState
from tools.jd_parser import extract_skills_offline, extract_requirements_llm
from tools.resume_matcher import match_resume_to_jd
from tools.gap_analyzer import analyze_gap, prioritize_gaps
from tools.bullet_writer import rewrite_bullet


def _with_retry(fn, *args, retries: int = 2, delay_seconds: float = 1.5, **kwargs):
    """Run fn with a small number of retries. Raises the last exception if all attempts fail."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay_seconds)
    raise last_exc


def run_pipeline(resume_text: str, jd_text: str, llm_backend: str = "ollama", anthropic_client=None) -> AgentState:
    state = AgentState(resume_text=resume_text, jd_text=jd_text)

    # --- Step 1: JD requirement extraction ---
    # Always run the offline pass first — it's free and instant, and becomes
    # our fallback if the LLM step fails.
    offline_result = extract_skills_offline(jd_text)
    state.jd_skills_offline = offline_result.matched_skills

    try:
        llm_result = _with_retry(
            extract_requirements_llm, jd_text, backend=llm_backend, client=anthropic_client, retries=2
        )
        state.jd_requirements = llm_result
        state.log("jd_extraction", "ok", f"LLM extraction succeeded via {llm_backend}")
    except Exception as e:
        state.jd_requirements = None
        state.log("jd_extraction", "degraded", f"LLM extraction failed ({e}); using offline skill match only")

    # --- Step 2: Resume-JD semantic matching ---
    try:
        match = match_resume_to_jd(resume_text, jd_text, jd_skills=state.jd_skills_offline)
        state.resume_match = {
            "jd_similarity_score": match.jd_similarity_score,
            "top_bullets": match.top_bullets,
            "skill_evidence": match.skill_evidence,
        }
        state.log("resume_matching", "ok")
    except Exception as e:
        state.resume_match = None
        state.log("resume_matching", "failed", str(e))

    # --- Step 3: Gap analysis (pure logic, should basically never fail) ---
    try:
        gap = analyze_gap(resume_text, jd_text)
        must_haves = (state.jd_requirements or {}).get("must_have_skills")
        state.gap_analysis = {
            "matched_skills": gap.matched_skills,
            "missing_skills": gap.missing_skills,
            "bonus_skills": gap.bonus_skills,
            "coverage_pct": gap.coverage_pct,
        }
        state.prioritized_gaps = prioritize_gaps(gap, must_have_skills=must_haves)
        state.log("gap_analysis", "ok")
    except Exception as e:
        state.log("gap_analysis", "failed", str(e))

    # --- Step 4: Bullet rewriting ---
    # Only rewrite the top bullets from Step 2 (the ones most relevant to this JD) —
    # no point spending LLM calls rewriting irrelevant bullets.
    top_bullets = (state.resume_match or {}).get("top_bullets", [])
    jd_skill_list = state.jd_skills_offline
    for bullet in top_bullets[:3]:
        try:
            rewritten = _with_retry(
                rewrite_bullet, bullet, jd_skill_list, backend=llm_backend, client=anthropic_client, retries=2
            )
            state.rewritten_bullets.append({"original": bullet, **rewritten})
        except Exception as e:
            # degrade to "keep original" rather than dropping the bullet entirely
            state.rewritten_bullets.append({
                "original": bullet,
                "rewritten_bullet": bullet,
                "keywords_incorporated": [],
                "confidence_note": f"Rewrite failed, kept original: {e}",
            })
    state.log(
        "bullet_rewriting",
        "ok" if top_bullets else "degraded",
        "" if top_bullets else "No resume_match available to select bullets from",
    )

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