"""
Tool 3: Gap Analyzer
Pure logic, no LLM/API dependency — runs instantly, zero cost, always available
even if Ollama isn't running. Diffs the skills a JD asks for against the skills
detectable in the resume, and produces a prioritized gap list.
"""

import re
from dataclasses import dataclass, field

from tools.jd_parser import PREFERRED, REQUIRED, extract_skills_offline
from tools.skill_taxonomy import canonicalize, group_by_category


@dataclass
class GapAnalysis:
    jd_skills: list[str]
    resume_skills: list[str]
    matched_skills: list[str] = field(default_factory=list)   # JD wants it, resume has it
    missing_skills: list[str] = field(default_factory=list)   # JD wants it, resume doesn't show it
    bonus_skills: list[str] = field(default_factory=list)     # resume has it, JD didn't ask
    coverage_pct: float = 0.0                                  # % of JD skills covered
    requirement_levels: dict[str, str] = field(default_factory=dict)  # JD skill -> required/preferred/mentioned

    def to_dict(self) -> dict:
        return {
            "jd_skills": self.jd_skills,
            "resume_skills": self.resume_skills,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "bonus_skills": self.bonus_skills,
            "coverage_pct": self.coverage_pct,
            "requirement_levels": self.requirement_levels,
            "missing_by_category": group_by_category(self.missing_skills),
            "matched_by_category": group_by_category(self.matched_skills),
        }


def _mentions(text_lower: str, phrase: str) -> bool:
    """Every significant word of a free-text skill appears in the text (word-boundary)."""
    words = [w for w in re.findall(r"[a-z0-9+#./-]+", phrase.lower()) if len(w) > 1 or w in {"r", "c"}]
    return bool(words) and all(
        re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", text_lower) for w in words
    )


def analyze_gap(resume_text: str, jd_text: str, extra_jd_skills: list[str] | None = None) -> GapAnalysis:
    """
    extra_jd_skills: optional free-text skills (typically the LLM's must-haves)
    that the taxonomy doesn't know about. Short ones (<= 3 words) are folded
    into the diff using a plain text match against the resume.
    """
    jd_result = extract_skills_offline(jd_text)
    resume_result = extract_skills_offline(resume_text)  # same taxonomy match, applied to resume text

    jd_set = set(jd_result.matched_skills)
    resume_set = set(resume_result.matched_skills)
    levels = dict(jd_result.requirement_levels)

    resume_lower = resume_text.lower()
    for raw in extra_jd_skills or []:
        raw = raw.strip()
        if not raw or len(raw.split()) > 3 or canonicalize(raw):
            continue  # empty, a sentence rather than a skill, or already covered by the taxonomy
        name = raw.lower()
        jd_set.add(name)
        levels.setdefault(name, REQUIRED)
        if _mentions(resume_lower, name):
            resume_set.add(name)

    matched = sorted(jd_set & resume_set)
    missing = sorted(jd_set - resume_set)
    bonus = sorted(resume_set - jd_set)

    coverage = round(len(matched) / len(jd_set) * 100, 1) if jd_set else 0.0

    return GapAnalysis(
        jd_skills=sorted(jd_set),
        resume_skills=sorted(resume_set),
        matched_skills=matched,
        missing_skills=missing,
        bonus_skills=bonus,
        coverage_pct=coverage,
        requirement_levels={k: v for k, v in levels.items() if k in jd_set},
    )


def _canonical_set(skills: list[str] | None) -> set[str]:
    """LLM skill names ("PyTorch", "K8s", "REST APIs") -> canonical taxonomy names, plus their lowercase form."""
    out = set()
    for skill in skills or []:
        out.add(skill.strip().lower())
        out.update(canonicalize(skill))
    return out


_PRIORITY_ORDER = {"critical": 0, "high": 1, "moderate": 2, "low": 3}


def prioritize_gaps(
    gap: GapAnalysis,
    must_have_skills: list[str] | None = None,
    nice_to_have_skills: list[str] | None = None,
    requirement_levels: dict[str, str] | None = None,
) -> list[dict]:
    """
    Rank missing skills by priority:
      critical — the LLM-based JD extraction lists it as a must-have
      high     — offline section detection found it under a "requirements" part of the JD
      moderate — mentioned in the JD, no clear signal either way
      low      — listed as a nice-to-have / bonus / "a plus"
    Without must_have_skills (pure offline mode) nothing is ever marked critical,
    and without requirement_levels every missing skill is moderate.
    """
    must_have_set = _canonical_set(must_have_skills)
    nice_set = _canonical_set(nice_to_have_skills)
    levels = requirement_levels or {}

    ranked = []
    for skill in gap.missing_skills:
        if skill in must_have_set:
            priority, reason = "critical", "Listed as a must-have"
        elif skill in nice_set:
            priority, reason = "low", "Listed as nice-to-have"
        elif levels.get(skill) == REQUIRED:
            priority, reason = "high", "Appears in the JD's requirements"
        elif levels.get(skill) == PREFERRED:
            priority, reason = "low", "Described as preferred / a plus"
        else:
            priority, reason = "moderate", "Mentioned in the JD"
        ranked.append({"skill": skill, "priority": priority, "reason": reason})
    ranked.sort(key=lambda x: (_PRIORITY_ORDER[x["priority"]], x["skill"]))
    return ranked


if __name__ == "__main__":
    sample_resume = """
    Built a HybridRAG customer complaint classifier using FAISS, Ollama, and Streamlit.
    Developed HoopSense AI, a full-stack basketball shooting analysis platform using React, FastAPI, MediaPipe, YOLOv8, and XGBoost.
    Built a resume parser and job matcher using NLP techniques and Scikit-learn.
    Implemented an ETL pipeline with PySpark and Databricks following medallion architecture.
    """
    sample_jd = """
    Machine Learning Engineering Intern - Summer 2027
    You'll work with PyTorch, build RAG pipelines using LangChain and FAISS,
    and deploy models via FastAPI and Docker on AWS. SQL and scikit-learn experience preferred.
    """

    gap = analyze_gap(sample_resume, sample_jd)
    print(f"JD asks for: {gap.jd_skills}")
    print(f"Resume shows: {gap.resume_skills}")
    print(f"\nMatched: {gap.matched_skills}")
    print(f"Missing: {gap.missing_skills}")
    print(f"Bonus (resume has, JD didn't ask): {gap.bonus_skills}")
    print(f"\nCoverage: {gap.coverage_pct}%")

    print("\nPrioritized gaps:")
    for item in prioritize_gaps(gap, requirement_levels=gap.requirement_levels):
        print(f"  [{item['priority']}] {item['skill']} — {item['reason']}")
