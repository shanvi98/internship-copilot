"""
Tests for tools/gap_analyzer.py.

gap_analyzer is pure logic layered on top of extract_skills_offline's fixed
taxonomy — no LLM, no embeddings, no network. That makes it the cheapest and
most reliable place in the whole pipeline to test, and the right place to
start since it needs zero mocking.

Run with:  pytest backend/tests/test_gap_analyzer.py -v
(run from the backend/ directory, or with backend/ on PYTHONPATH)
"""

import pytest
from tools.gap_analyzer import analyze_gap, prioritize_gaps


def test_full_overlap_gives_100_percent_coverage():
    """If the resume mentions every skill the JD asks for, coverage should be 100%."""
    jd = "Looking for someone with Python and Docker experience."
    resume = "I have built projects using Python and Docker extensively."

    gap = analyze_gap(resume, jd)

    assert gap.coverage_pct == 100.0
    assert gap.missing_skills == []
    assert set(gap.matched_skills) == {"python", "docker"}


def test_no_overlap_gives_zero_percent_coverage():
    """If the resume shares none of the JD's skills, coverage should be 0%, not crash."""
    jd = "Looking for someone with Kubernetes and GraphQL experience."
    resume = "I mostly write marketing copy and manage social media."

    gap = analyze_gap(resume, jd)

    assert gap.coverage_pct == 0.0
    assert set(gap.missing_skills) == {"kubernetes", "graphql"}
    assert gap.matched_skills == []


def test_bonus_skills_are_resume_skills_not_asked_for():
    """Skills the resume has that the JD never mentioned should land in bonus, not missing."""
    jd = "Looking for someone with Python experience."
    resume = "I use Python, PyTorch, and Docker in my projects."

    gap = analyze_gap(resume, jd)

    assert "python" in gap.matched_skills
    assert set(gap.bonus_skills) == {"pytorch", "docker"}
    assert gap.missing_skills == []


def test_empty_jd_skill_set_gives_zero_coverage_not_division_error():
    """
    Edge case: if the JD text contains none of our taxonomy's skills (e.g. a
    very generic or malformed JD), coverage_pct must not raise a ZeroDivisionError.
    """
    jd = "We are looking for a wonderful, dedicated team player."
    resume = "I have experience with Python and SQL."

    gap = analyze_gap(resume, jd)

    assert gap.jd_skills == []
    assert gap.coverage_pct == 0.0


def test_partial_overlap_computes_correct_percentage():
    """Coverage should be an accurate percentage, not just present/absent."""
    jd = "Requires Python, SQL, Docker, and AWS."  # 4 skills
    resume = "I know Python and SQL well."  # matches 2 of 4

    gap = analyze_gap(resume, jd)

    assert gap.coverage_pct == 50.0
    assert set(gap.matched_skills) == {"python", "sql"}
    assert set(gap.missing_skills) == {"docker", "aws"}


def test_prioritize_gaps_ranks_must_haves_as_critical():
    """Missing skills that are in the LLM-derived must_have list should be flagged critical
    and sorted ahead of moderate-priority missing skills."""
    jd = "Requires Python, Docker, and Kubernetes."
    resume = "I only know Python."

    gap = analyze_gap(resume, jd)
    ranked = prioritize_gaps(gap, must_have_skills=["docker"])

    priorities = {item["skill"]: item["priority"] for item in ranked}
    assert priorities["docker"] == "critical"
    assert priorities["kubernetes"] == "moderate"
    # critical items must be sorted first
    assert ranked[0]["priority"] == "critical"


def test_prioritize_gaps_without_must_haves_treats_all_as_moderate():
    """Pure offline mode (no LLM extraction available): every missing skill is 'moderate',
    never silently marked critical."""
    jd = "Requires Python and Docker."
    resume = "I don't know either."

    gap = analyze_gap(resume, jd)
    ranked = prioritize_gaps(gap)  # no must_have_skills passed

    assert all(item["priority"] == "moderate" for item in ranked)


def test_empty_resume_does_not_crash():
    """An empty or whitespace-only resume should still return a valid GapAnalysis,
    not raise — this is the kind of input a real user WILL submit accidentally."""
    jd = "Requires Python and SQL."
    resume = "   "

    gap = analyze_gap(resume, jd)

    assert gap.matched_skills == []
    assert set(gap.missing_skills) == {"python", "sql"}