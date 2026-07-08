"""
Tool 3: Gap Analyzer
Pure logic, no LLM/API dependency — runs instantly, zero cost, always available
even if Ollama isn't running. Diffs the skills a JD asks for against the skills
detectable in the resume, and produces a prioritized gap list.
"""

from dataclasses import dataclass, field
from tools.jd_parser import extract_skills_offline


@dataclass
class GapAnalysis:
    jd_skills: list[str]
    resume_skills: list[str]
    matched_skills: list[str] = field(default_factory=list)   # JD wants it, resume has it
    missing_skills: list[str] = field(default_factory=list)   # JD wants it, resume doesn't show it
    bonus_skills: list[str] = field(default_factory=list)     # resume has it, JD didn't ask
    coverage_pct: float = 0.0                                  # % of JD skills covered


def analyze_gap(resume_text: str, jd_text: str) -> GapAnalysis:
    jd_result = extract_skills_offline(jd_text)
    resume_result = extract_skills_offline(resume_text)  # same taxonomy match, applied to resume text

    jd_set = set(jd_result.matched_skills)
    resume_set = set(resume_result.matched_skills)

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
    )


def prioritize_gaps(gap: GapAnalysis, must_have_skills: list[str] | None = None) -> list[dict]:
    """
    Rank missing skills by priority. If must_have_skills is provided (from the
    LLM-based JD extraction), missing must-haves are flagged critical.
    Without it (pure offline mode), every missing skill is treated as equal priority.
    """
    must_have_set = set(must_have_skills or [])
    ranked = []
    for skill in gap.missing_skills:
        priority = "critical" if skill in must_have_set else "moderate"
        ranked.append({"skill": skill, "priority": priority})
    # critical first
    ranked.sort(key=lambda x: x["priority"] != "critical")
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
    for item in prioritize_gaps(gap):
        print(f"  [{item['priority']}] {item['skill']}")
