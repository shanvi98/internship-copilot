"""
Tool 2: Resume Matcher
Given resume text and JD text, computes semantic similarity per resume "chunk"
(bullet point / project description) against the JD as a whole, and against
each matched skill. Uses sentence-transformers so it runs fully offline —
no API cost for what will likely be the most frequently called tool.
"""

from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util

_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this task
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


@dataclass
class MatchResult:
    jd_similarity_score: float  # 0-1, overall resume-vs-JD semantic fit
    bullet_scores: list[tuple[str, float]]  # each resume bullet + its similarity to the JD
    top_bullets: list[str]  # bullets most relevant to this JD, ranked


def split_resume_into_bullets(resume_text: str) -> list[str]:
    """Split resume into individual lines/bullets, dropping empty lines and headers."""
    lines = [l.strip("•-*  \t") for l in resume_text.split("\n")]
    # keep lines with enough words to be a real bullet, not a section header like "SKILLS"
    return [l for l in lines if len(l.split()) >= 4]


def match_resume_to_jd(resume_text: str, jd_text: str, top_k: int = 5) -> MatchResult:
    model = get_model()
    bullets = split_resume_into_bullets(resume_text)
    if not bullets:
        raise ValueError("No resume bullets found — check resume_text formatting")

    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    bullet_embeddings = model.encode(bullets, convert_to_tensor=True)

    similarities = util.cos_sim(bullet_embeddings, jd_embedding).squeeze(1).tolist()
    bullet_scores = list(zip(bullets, similarities))
    bullet_scores.sort(key=lambda x: x[1], reverse=True)

    overall_score = sum(s for _, s in bullet_scores) / len(bullet_scores)

    return MatchResult(
        jd_similarity_score=round(overall_score, 3),
        bullet_scores=[(b, round(s, 3)) for b, s in bullet_scores],
        top_bullets=[b for b, _ in bullet_scores[:top_k]],
    )


if __name__ == "__main__":
    sample_resume = """
    SKILLS
    Built a HybridRAG customer complaint classifier using FAISS, Ollama, and Streamlit.
    Developed HoopSense AI, a full-stack basketball shooting analysis platform using React, FastAPI, MediaPipe, YOLOv8, and XGBoost.
    Built a resume parser and job matcher using NLP techniques and Scikit-learn.
    Created a food outlet website using HTML CSS and JavaScript for a local business.
    Implemented an ETL pipeline with PySpark and Databricks following medallion architecture.
    """
    sample_jd = """
    Machine Learning Engineering Intern - Summer 2027
    You'll work with PyTorch, build RAG pipelines using LangChain and FAISS,
    and deploy models via FastAPI and Docker on AWS.
    """
    result = match_resume_to_jd(sample_resume, sample_jd)
    print(f"Overall JD fit score: {result.jd_similarity_score}\n")
    print("Ranked bullets (most relevant first):")
    for bullet, score in result.bullet_scores:
        print(f"  [{score:.3f}] {bullet[:80]}")
