"""
Tool 2: Resume Matcher
Given resume text and JD text, computes semantic similarity per resume "chunk"
(bullet point / project description) against the JD as a whole, and — when a
skill list is supplied — identifies which bullet best evidences each skill.
Uses sentence-transformers so it runs fully offline — no API cost for what
will likely be the most frequently called tool.

sentence-transformers (and torch) are imported lazily, the first time
embeddings are needed: the rest of the app (offline analysis, tests, the API
itself) doesn't pay a multi-second import or need the package installed at all.
"""

import re
import threading
from dataclasses import dataclass, field

from tools.skill_taxonomy import find_skills

_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this task
_model = None
_model_lock = threading.Lock()

# Below this cosine similarity, we don't consider a bullet meaningful evidence
# for a skill — better to say "no evidence found" than to force a weak match.
_SKILL_EVIDENCE_THRESHOLD = 0.40

# MiniLM truncates input at 256 word-pieces; a full JD is far longer, so the JD
# is embedded chunk-by-chunk and mean-pooled instead of silently truncated.
_JD_CHUNK_MIN_WORDS = 4

# Share of a bullet's relevance score that comes from embeddings (the rest from skill overlap).
_SEMANTIC_WEIGHT = 0.6

_BULLET_GLYPHS = "•●▪◦‣⁃∙·–—-*>  \t"
_NUMBERED = re.compile(r"^\(?\d{1,2}[.)]\s+")
_CONTACT_LINE = re.compile(r"@|https?://|linkedin\.com|github\.com|\+?\d[\d\s().-]{8,}\d")
# Degree / school lines ("B.S. Computer Science, State University") aren't accomplishments.
_EDUCATION_LINE = re.compile(
    r"\b(?:B\.?S\.?c?|B\.?A\.?|M\.?S\.?c?|M\.?Eng|Ph\.?D|Bachelor(?:'s)?|Master(?:'s)?|GPA|University|College|"
    r"Institute of|High School)\b"
)


def get_model():
    global _model
    if _model is None:
        with _model_lock:  # FastAPI runs sync endpoints in a thread pool
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(_MODEL_NAME)
    return _model


def is_model_loaded() -> bool:
    return _model is not None


@dataclass
class MatchResult:
    jd_similarity_score: float  # 0-1, semantic fit of the resume's strongest bullets to the JD
    bullet_scores: list[tuple[str, float]]  # each resume bullet + its relevance to the JD (semantic + skill overlap)
    top_bullets: list[str]  # bullets most relevant to this JD, ranked
    skill_evidence: dict[str, str | None] = field(default_factory=dict)  # skill -> best supporting bullet, or None


def split_resume_into_bullets(resume_text: str) -> list[str]:
    """Split resume into individual bullets, dropping headers, contact lines and blank lines."""
    bullets = []
    for raw in resume_text.replace("\r", "").split("\n"):
        line = _NUMBERED.sub("", raw.strip().strip(_BULLET_GLYPHS)).strip()
        words = len(line.split())
        if words < 4 or (words < 12 and (_CONTACT_LINE.search(line) or _EDUCATION_LINE.search(line))):
            continue  # section headers ("SKILLS"), contact info, degree lines, dates
        if len(line.split()) > 45:
            # A pasted paragraph (or a PDF that lost its line breaks): split on sentences.
            bullets.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if len(s.split()) >= 4)
        else:
            bullets.append(line)
    # de-duplicate while preserving order (PDF extraction sometimes repeats lines)
    return list(dict.fromkeys(bullets))


def _jd_chunks(jd_text: str) -> list[str]:
    chunks = [c.strip() for c in re.split(r"\n+|(?<=[.!?])\s+", jd_text) if len(c.split()) >= _JD_CHUNK_MIN_WORDS]
    return chunks or [jd_text]


def match_resume_to_jd(
    resume_text: str, jd_text: str, jd_skills: list[str] | None = None, top_k: int = 5
) -> MatchResult:
    """
    jd_skills is optional: pass the JD's matched skill list (e.g. from
    extract_skills_offline) to also get a per-skill "which bullet proves this"
    mapping. Omit it and you just get the overall JD-fit ranking.
    """
    from sentence_transformers import util

    bullets = split_resume_into_bullets(resume_text)
    if not bullets:
        raise ValueError("No resume bullets found — check resume_text formatting")

    model = get_model()
    chunk_embeddings = model.encode(_jd_chunks(jd_text), convert_to_tensor=True, normalize_embeddings=True)
    jd_embedding = chunk_embeddings.mean(dim=0, keepdim=True)
    bullet_embeddings = model.encode(bullets, convert_to_tensor=True, normalize_embeddings=True)

    similarities = util.cos_sim(bullet_embeddings, jd_embedding).squeeze(1).tolist()

    # Score the resume on its strongest bullets: a long resume with many
    # unrelated bullets shouldn't be penalized for having them.
    best = sorted(similarities, reverse=True)[:top_k]
    overall_score = sum(best) / len(best)

    # Rank bullets by a blend of semantic similarity and literal JD-skill overlap.
    # A small embedding model alone over-weights surface words (an "ETL pipeline"
    # bullet can outrank a "RAG with FAISS" bullet for a RAG job just on "pipeline").
    bullet_skills = [set(find_skills(b)) for b in bullets]
    jd_skill_set = set(jd_skills or [])
    if jd_skill_set:
        denom = min(3, len(jd_skill_set))
        relevance = [
            _SEMANTIC_WEIGHT * sim + (1 - _SEMANTIC_WEIGHT) * min(1.0, len(skills & jd_skill_set) / denom)
            for sim, skills in zip(similarities, bullet_skills)
        ]
    else:
        relevance = similarities
    bullet_scores = sorted(zip(bullets, relevance), key=lambda x: x[1], reverse=True)

    skill_evidence: dict[str, str | None] = {}
    if jd_skills:
        # Literal mentions are the most convincing evidence; embeddings are the fallback
        # for skills the bullet describes without naming.
        # Skills that only appear in a short "SKILLS: Python, SQL" line aren't bullets, but that
        # line is still the honest evidence for them.
        other_lines = [l.strip(_BULLET_GLYPHS) for l in resume_text.splitlines() if l.strip() and l.strip(_BULLET_GLYPHS) not in bullets]
        candidates = list(zip(bullets, bullet_skills)) + [(l, set(find_skills(l))) for l in other_lines]
        need_semantic = []
        for skill in jd_skills:
            literal = next((line for line, skills in candidates if skill in skills), None)
            if literal is not None:
                skill_evidence[skill] = literal
            else:
                need_semantic.append(skill)

        if need_semantic:
            skill_embeddings = model.encode(need_semantic, convert_to_tensor=True, normalize_embeddings=True)
            # one bullets-vs-skills matrix, reused for every skill — cheaper than
            # re-encoding bullets once per skill
            skill_sims = util.cos_sim(bullet_embeddings, skill_embeddings)  # [num_bullets x num_skills]
            for skill_idx, skill in enumerate(need_semantic):
                col = skill_sims[:, skill_idx].tolist()
                best_bullet_idx = max(range(len(col)), key=lambda i: col[i])
                skill_evidence[skill] = (
                    bullets[best_bullet_idx] if col[best_bullet_idx] >= _SKILL_EVIDENCE_THRESHOLD else None
                )

    return MatchResult(
        jd_similarity_score=round(overall_score, 3),
        bullet_scores=[(b, round(s, 3)) for b, s in bullet_scores],
        top_bullets=[b for b, _ in bullet_scores[:top_k]],
        skill_evidence=skill_evidence,
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
