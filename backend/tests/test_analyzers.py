"""Tests for the deterministic analyzers: ATS keywords, resume linter, match score."""

from tools.keyword_analyzer import analyze_keywords
from tools.resume_linter import lint_resume
from tools.resume_matcher import split_resume_into_bullets
from tools.scoring import compute_match_score, weighted_skill_coverage


# --- keyword analyzer ---

def test_keywords_match_plural_and_verb_forms():
    jd = "You will build dashboards and forecasting models for the payments team."
    result = analyze_keywords("Built a dashboard and a demand forecasting model.", jd)
    found = {k["term"]: k["in_resume"] for k in result["keywords"]}

    assert found["dashboards"] is True
    assert found["forecasting"] is True
    assert found["payments"] is False
    assert "payments" in result["missing_keywords"]


def test_keywords_ignore_benefits_boilerplate():
    jd = "Requirements:\n- Fraud detection with Python\nBenefits\nFree lunch and gym stipend\nGenerous PTO\n"
    terms = {k["term"] for k in analyze_keywords("Python", jd)["keywords"]}

    assert {"fraud", "detection", "python"} <= terms
    assert not terms & {"lunch", "gym", "stipend", "generous", "pto"}


def test_keywords_handle_empty_jd():
    result = analyze_keywords("Python developer", "")
    assert result["keywords"] == []
    assert result["keyword_match_pct"] == 0.0


# --- resume linter ---

def test_linter_flags_weak_openers_and_first_person(sample_resume):
    issues = lint_resume(sample_resume + "\n• I helped my team ship a new onboarding flow quickly.\n")["issues"]
    types = [i["type"] for i in issues]

    assert "weak_opener" in types
    assert "first_person" in types
    weak = [i for i in issues if i["type"] == "weak_opener"]
    assert any("Responsible for" in i["text"] for i in weak)


def test_linter_reports_contact_and_sections(sample_resume):
    report = lint_resume(sample_resume)
    assert report["has_email"] is True
    assert {"education", "projects", "skills"} <= set(report["sections_found"])
    assert not any(i["type"] == "missing_contact" for i in report["issues"])


def test_linter_flags_missing_email_and_metrics():
    resume = "Built a web scraper using Python and requests library.\nDesigned a REST API with Flask and SQLite.\n"
    types = {i["type"] for i in lint_resume(resume)["issues"]}
    assert {"missing_contact", "few_metrics", "missing_sections"} <= types


# --- bullet splitting ---

def test_split_resume_drops_headers_and_contact_lines(sample_resume):
    bullets = split_resume_into_bullets(sample_resume)
    assert all(not b.startswith(("•", "-")) for b in bullets)
    assert not any("alex@example.com" in b for b in bullets)
    assert not any(b in ("EDUCATION", "SKILLS") for b in bullets)
    assert not any("State University" in b for b in bullets)
    assert len(bullets) == 3


def test_split_resume_breaks_long_paragraphs_into_sentences():
    paragraph = " ".join(["Built a large data pipeline in Python for analytics workloads."] * 6)
    assert len(split_resume_into_bullets(paragraph)) == 1  # identical sentences are de-duplicated
    varied = " ".join(f"Built pipeline number {i} using Python and Airflow for team {i}." for i in range(6))
    assert len(split_resume_into_bullets(varied)) == 6


# --- match score ---

def test_required_skills_weigh_more_than_preferred():
    levels = {"python": "required", "docker": "preferred"}
    assert weighted_skill_coverage(["python"], ["python", "docker"], levels) == 80.0
    assert weighted_skill_coverage(["docker"], ["python", "docker"], levels) == 20.0


def test_match_score_renormalizes_over_available_signals():
    gap = {"jd_skills": ["python", "sql"], "matched_skills": ["python", "sql"], "requirement_levels": {}}
    only_skills = compute_match_score(gap)
    assert only_skills["score"] == 100
    assert only_skills["label"] == "strong"
    assert set(only_skills["components"]) == {"skills"}

    blended = compute_match_score(gap, keyword_match_pct=0.0, semantic_similarity=0.15)
    assert blended["score"] == 55
    assert set(blended["components"]) == {"skills", "keywords", "semantic"}


def test_match_score_with_no_signals():
    assert compute_match_score({"jd_skills": []})["label"] == "unknown"


def test_split_resume_drops_degree_lines():
    resume = "B.S. Computer Science, State University, GPA 3.8\nBuilt a REST API with Flask for course registration."
    assert split_resume_into_bullets(resume) == ["Built a REST API with Flask for course registration."]
