"""
Tests for the skill taxonomy and offline JD parsing (tools/skill_taxonomy.py, tools/jd_parser.py).

These guard the false positives / false negatives that make gap analysis
wrong: a skill the JD asks for under a different spelling showing up as
"missing", or ordinary English words being counted as skills.
"""

import pytest

from tools.jd_parser import PREFERRED, REQUIRED, extract_skills_offline
from tools.skill_taxonomy import canonicalize, find_skills, group_by_category


@pytest.mark.parametrize("text, expected", [
    ("Experience with LLMs", "llm"),
    ("YOLOv8 object detection", "yolo"),
    ("sklearn pipelines", "scikit-learn"),
    ("deployed on k8s", "kubernetes"),
    ("RESTful APIs", "rest api"),
    ("CI / CD pipelines", "ci/cd"),
    ("Hugging Face models", "huggingface"),
    ("Postgres database", "postgresql"),
    ("Node.js backend", "node.js"),
    ("Built HybridRAG with FAISS", "rag"),
    ("proficient in Go", "go"),
    ("Languages: Python, R", "r"),
])
def test_aliases_map_to_canonical_skill(text, expected):
    assert expected in find_skills(text)


@pytest.mark.parametrize("text, not_expected", [
    ("Our R&D team", "r"),
    ("We go to market quickly", "go"),
    ("You will excel at communication", "excel"),
    ("Collaborate with the rest of the team", "rest api"),
    ("JavaScript developer", "java"),
    ("drag and drop builder", "rag"),
    ("PySpark jobs", "python"),
    ("github.com/me", "github actions"),
])
def test_common_words_are_not_skills(text, not_expected):
    assert not_expected not in find_skills(text)


def test_postgres_implies_sql():
    """A resume listing PostgreSQL shouldn't be told it's missing SQL."""
    assert {"postgresql", "sql"} <= set(find_skills("PostgreSQL"))


def test_canonicalize_llm_skill_names():
    assert canonicalize("PyTorch") == ["pytorch"]
    assert canonicalize("Experience with K8s") == ["kubernetes"]
    assert canonicalize("Strong communication skills") == []


def test_group_by_category():
    groups = group_by_category(["python", "docker", "pytorch", "some-unknown-skill"])
    assert groups["languages"] == ["python"]
    assert groups["cloud_devops"] == ["docker"]
    assert groups["other"] == ["some-unknown-skill"]


@pytest.mark.parametrize("text, expected", [
    ("Software Engineering Intern", "intern"),
    ("Our internal tools team is hiring a Senior Engineer", "senior"),
    ("International company seeks new grad engineers", "new grad"),
    ("Staffing agency placing analysts", None),
])
def test_seniority_uses_word_boundaries(text, expected):
    assert extract_skills_offline(text).seniority_hint == expected


def test_role_title_is_first_line_naming_a_role(sample_jd):
    assert extract_skills_offline(sample_jd).role_title == "Machine Learning Engineering Intern - Summer 2027"


def test_requirement_levels_follow_sections(sample_jd):
    levels = extract_skills_offline(sample_jd).requirement_levels
    assert levels["python"] == REQUIRED
    assert levels["docker"] == REQUIRED
    assert levels["kubernetes"] == PREFERRED


def test_inline_cues_classify_individual_sentences():
    jd = "Python is required. Experience with Docker is a plus."
    levels = extract_skills_offline(jd).requirement_levels
    assert levels["python"] == REQUIRED
    assert levels["docker"] == PREFERRED
