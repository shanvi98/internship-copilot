"""
Tests for agent/orchestrator.py: the retry / degrade / circuit-breaker behavior
that makes the pipeline an agent rather than a script. All tools that would
need an LLM or the embedding model are monkeypatched.
"""

import pytest

from agent import orchestrator
from tools.bullet_writer import UnsupportedClaimError
from tools.llm import LLMUnavailable


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(orchestrator.time, "sleep", lambda s: None)


def _statuses(state):
    return {log.step_name: log.status for log in state.logs}


def test_llm_unavailable_skips_rewrites_without_retrying(monkeypatch, sample_resume, sample_jd):
    calls = {"extract": 0, "rewrite": 0}

    def extract(*a, **k):
        calls["extract"] += 1
        raise LLMUnavailable("Ollama is not reachable")

    def rewrite(*a, **k):
        calls["rewrite"] += 1
        return {}

    monkeypatch.setattr(orchestrator, "extract_requirements_llm", extract)
    monkeypatch.setattr(orchestrator, "rewrite_bullet", rewrite)

    state = orchestrator.run_pipeline(sample_resume, sample_jd, use_embeddings=False)

    assert calls == {"extract": 1, "rewrite": 0}  # no retry, and no doomed rewrite calls
    assert _statuses(state) == {
        "jd_extraction": "degraded",
        "resume_matching": "skipped",
        "gap_analysis": "ok",
        "bullet_rewriting": "skipped",
    }
    summary = state.summary()
    assert summary["gap_analysis"]["coverage_pct"] > 0
    assert summary["match_score"]["score"] > 0
    assert summary["prioritized_gaps"]


def test_successful_llm_run_marks_must_haves_critical(monkeypatch, sample_resume, sample_jd):
    monkeypatch.setattr(orchestrator, "extract_requirements_llm", lambda *a, **k: {
        "role_title": "ML Intern", "company": "Acme", "must_have_skills": ["PyTorch", "Docker"],
        "nice_to_have_skills": ["Kubernetes"], "seniority": "intern", "key_responsibilities": [],
    })
    monkeypatch.setattr(orchestrator, "rewrite_bullet", lambda bullet, skills, **k: {
        "rewritten_bullet": bullet + " (tailored)", "keywords_incorporated": [], "confidence_note": "",
    })

    state = orchestrator.run_pipeline(sample_resume, sample_jd, use_embeddings=False, max_rewrites=2)
    summary = state.summary()

    assert set(_statuses(state).values()) <= {"ok", "skipped"}
    assert summary["jd_overview"]["company"] == "Acme"
    priorities = {g["skill"]: g["priority"] for g in summary["prioritized_gaps"]}
    assert priorities["pytorch"] == "critical"
    assert priorities["docker"] == "critical"
    assert priorities["kubernetes"] == "low"
    assert len(summary["rewritten_bullets"]) == 2
    assert all(b["status"] == "ok" for b in summary["rewritten_bullets"])


def test_partial_rewrite_failure_is_degraded_and_keeps_original(monkeypatch, sample_resume, sample_jd):
    monkeypatch.setattr(orchestrator, "extract_requirements_llm", lambda *a, **k: {"must_have_skills": []})
    attempts = {}

    def rewrite(bullet, skills, **k):
        attempts[bullet] = attempts.get(bullet, 0) + 1
        if "HybridRAG" in bullet:
            raise UnsupportedClaimError("added numbers not in the original: 99")
        return {"rewritten_bullet": "Rewritten", "keywords_incorporated": [], "confidence_note": ""}

    monkeypatch.setattr(orchestrator, "rewrite_bullet", rewrite)
    state = orchestrator.run_pipeline(sample_resume, sample_jd, use_embeddings=False, max_rewrites=3)

    assert _statuses(state)["bullet_rewriting"] == "degraded"
    failed = [b for b in state.rewritten_bullets if b["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["rewritten_bullet"] == failed[0]["original"]
    assert "kept your original" in failed[0]["confidence_note"]
    # a bad response is worth retrying once
    assert attempts[failed[0]["original"]] == 2


def test_embedding_failure_degrades_to_first_bullets(monkeypatch, sample_resume, sample_jd):
    monkeypatch.setattr(orchestrator, "extract_requirements_llm", lambda *a, **k: (_ for _ in ()).throw(LLMUnavailable("x")))

    def broken_match(*a, **k):
        raise ImportError("No module named 'sentence_transformers'")

    monkeypatch.setattr(orchestrator, "match_resume_to_jd", broken_match)
    state = orchestrator.run_pipeline(sample_resume, sample_jd, use_embeddings=True)

    assert _statuses(state)["resume_matching"] == "degraded"
    assert state.resume_match["top_bullets"]
    assert state.resume_match["jd_similarity_score"] is None


def test_offline_analysis_has_every_section(sample_resume, sample_jd):
    result = orchestrator.run_offline_analysis(sample_resume, sample_jd)
    assert set(result) == {"jd_overview", "gap_analysis", "prioritized_gaps", "keyword_analysis", "resume_checks", "match_score"}
    assert result["jd_overview"]["seniority"] == "intern"


def test_steps_record_durations(monkeypatch, sample_resume, sample_jd):
    monkeypatch.setattr(orchestrator, "extract_requirements_llm", lambda *a, **k: (_ for _ in ()).throw(LLMUnavailable("x")))
    steps = orchestrator.run_pipeline(sample_resume, sample_jd, use_embeddings=False).summary()["steps"]
    assert all(isinstance(s["duration_ms"], int) and s["duration_ms"] >= 0 for s in steps)
