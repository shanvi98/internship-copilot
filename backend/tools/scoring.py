"""
Overall match score: one 0-100 number that blends the separate signals the
pipeline produces, so a user comparing several postings can rank them.

  skill fit   — share of JD skills covered, with required skills weighted 2x
                and nice-to-haves 0.5x (from gap analysis)
  keywords    — share of the JD's prominent terms the resume uses (ATS view)
  semantic    — embedding similarity of the resume's best bullets to the JD,
                when the embedding model ran

Weights are re-normalized over the signals that are actually available, so the
offline-only path still produces a meaningful score.
"""

from tools.jd_parser import PREFERRED, REQUIRED

_LEVEL_WEIGHT = {REQUIRED: 2.0, PREFERRED: 0.5}
_WEIGHTS = {"skills": 0.55, "keywords": 0.25, "semantic": 0.20}

# MiniLM cosine similarities for resume-vs-JD text cluster roughly in [0.15, 0.65];
# rescale that band to 0-100 so the number is intuitive.
_SEM_LOW, _SEM_HIGH = 0.15, 0.65


def weighted_skill_coverage(matched: list[str], jd_skills: list[str], levels: dict[str, str]) -> float:
    if not jd_skills:
        return 0.0
    total = sum(_LEVEL_WEIGHT.get(levels.get(s, ""), 1.0) for s in jd_skills)
    got = sum(_LEVEL_WEIGHT.get(levels.get(s, ""), 1.0) for s in matched)
    return round(got / total * 100, 1)


def compute_match_score(
    gap: dict,
    keyword_match_pct: float | None = None,
    semantic_similarity: float | None = None,
) -> dict:
    components: dict[str, float] = {}
    if gap.get("jd_skills"):
        components["skills"] = weighted_skill_coverage(
            gap.get("matched_skills", []), gap["jd_skills"], gap.get("requirement_levels", {})
        )
    if keyword_match_pct is not None:
        components["keywords"] = float(keyword_match_pct)
    if semantic_similarity is not None:
        scaled = (semantic_similarity - _SEM_LOW) / (_SEM_HIGH - _SEM_LOW) * 100
        components["semantic"] = round(min(100.0, max(0.0, scaled)), 1)

    if not components:
        return {"score": 0, "label": "unknown", "components": {}}

    weight_sum = sum(_WEIGHTS[k] for k in components)
    score = round(sum(components[k] * _WEIGHTS[k] for k in components) / weight_sum)
    label = "strong" if score >= 70 else "fair" if score >= 45 else "weak"
    return {"score": score, "label": label, "components": components}
