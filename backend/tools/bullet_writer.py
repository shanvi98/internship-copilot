"""
Tool 4: Bullet Rewriter
Takes a resume bullet + the JD's key requirements, and rewrites the bullet to
mirror the JD's language (helps with ATS keyword matching) while staying
truthful to what the person actually did — this tool must NEVER invent
metrics or claims the person didn't provide.

The prompt asks the model not to invent anything, but prompts are requests,
not guarantees. verify_rewrite() checks the output deterministically: any
number or concrete tool that appears in the rewrite but not in the original
bullet makes the rewrite fail, so the orchestrator retries or keeps the original.

Default backend: Ollama (free, local). Optional: Anthropic API.
"""

import re

from schemas import BulletRewriteResult
from tools.llm import complete_json
from tools.skill_taxonomy import find_skills

REWRITE_PROMPT = """You are helping a student tailor ONE resume bullet point to better match a job description, for ATS keyword alignment.

STRICT RULES:
- Do NOT invent numbers, metrics, or outcomes that aren't in the original bullet.
- Do NOT claim tools/skills the original bullet doesn't mention.
- You MAY rephrase using synonyms/terms from the JD if they accurately describe what's already there (e.g. "built a REST API" -> "developed RESTful API endpoints" is fine; adding "achieved 99.9% uptime" when not stated is NOT fine).
- Keep it one line, resume-bullet style, starting with a strong past-tense action verb. No first-person pronouns.
- Return JSON only:
{{"rewritten_bullet": string, "keywords_incorporated": [string], "confidence_note": string}}

confidence_note should flag if you were unable to naturally incorporate JD keywords without overstating the original.

ORIGINAL BULLET:
{original_bullet}

JD KEY SKILLS/REQUIREMENTS:
{jd_skills}
"""

BULLET_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_bullet": {"type": "string"},
        "keywords_incorporated": {"type": "array", "items": {"type": "string"}},
        "confidence_note": {"type": "string"},
    },
    "required": ["rewritten_bullet", "keywords_incorporated", "confidence_note"],
    "additionalProperties": False,
}

# Broad concepts a rewrite may use to rephrase what's already described (e.g. a
# bullet about "a classifier using scikit-learn" can fairly say "machine learning").
# Anything else from the taxonomy — a concrete language, framework, cloud, tool —
# must already be in the original bullet.
_REPHRASABLE_CONCEPTS = {
    "machine learning", "deep learning", "nlp", "computer vision", "rest api", "microservices",
    "data pipeline", "etl", "model deployment", "llm", "generative ai", "ai agents",
    "data visualization", "oop", "algorithms", "data structures", "statistics",
}

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


class UnsupportedClaimError(ValueError):
    """The rewrite added a number or tool that the original bullet doesn't support."""


def verify_rewrite(original: str, rewritten: str) -> list[str]:
    """Return a list of human-readable problems; empty means the rewrite is safe."""
    problems = []
    orig_numbers = {n.replace(",", "") for n in _NUMBER.findall(original)}
    new_numbers = sorted({n.replace(",", "") for n in _NUMBER.findall(rewritten)} - orig_numbers)
    if new_numbers:
        problems.append(f"added numbers not in the original: {', '.join(new_numbers)}")

    new_skills = sorted(set(find_skills(rewritten)) - set(find_skills(original)) - _REPHRASABLE_CONCEPTS)
    if new_skills:
        problems.append(f"added skills not in the original: {', '.join(new_skills)}")
    return problems


def rewrite_bullet(
    original_bullet: str,
    jd_skills: list[str],
    backend: str = "ollama",
    model: str | None = None,
    client=None,
) -> dict:
    prompt = REWRITE_PROMPT.format(
        original_bullet=original_bullet,
        jd_skills=", ".join(jd_skills) or "(none detected)",
    )
    result = complete_json(prompt, BulletRewriteResult, BULLET_REWRITE_SCHEMA, backend=backend, model=model, client=client)

    problems = verify_rewrite(original_bullet, result.rewritten_bullet)
    if problems:
        raise UnsupportedClaimError("; ".join(problems))

    # Only report keywords that genuinely ended up in the text.
    lowered = result.rewritten_bullet.lower()
    result.keywords_incorporated = [k for k in result.keywords_incorporated if k.lower() in lowered]
    return result.model_dump()


if __name__ == "__main__":
    # This requires `ollama serve` running locally with a model pulled.
    import sys

    bullet = sys.argv[1] if len(sys.argv) > 1 else "Built a resume parser and job matcher using NLP techniques and Scikit-learn."
    try:
        print(rewrite_bullet(bullet, ["python", "scikit-learn", "nlp", "machine learning"]))
    except Exception as e:  # noqa: BLE001 — demo script
        print(f"Rewrite failed: {e}")
        print("Run `ollama serve` and `ollama pull llama3.1:8b`, then try again.")
