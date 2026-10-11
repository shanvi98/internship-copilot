"""
Tool 6: Resume Linter
Deterministic checks for the resume-writing mistakes recruiters and ATS
parsers penalize most often: weak openers ("Responsible for…"), unquantified
bullets, first-person pronouns, overly long bullets, and missing contact
info or standard sections. No LLM — every finding points at a specific line.
"""

import re

from tools.resume_matcher import split_resume_into_bullets

_WEAK_OPENERS = re.compile(
    r"^(responsible for|worked on|working on|helped(?: with| to)?|assisted(?: with| in)?|"
    r"involved in|participated in|tasked with|duties included|in charge of|was part of|did)\b",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(r"\b(?:I(?!/)|[Mm]e|[Mm]y|[Mm]ine|[Ww]e|[Oo]ur)\b")
_HAS_METRIC = re.compile(r"\d|%|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|dozens?|hundreds?|thousands?|millions?)\b", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+?\d[\s().-]*){10,}")
_LINK = re.compile(r"(linkedin\.com|github\.com|https?://)", re.IGNORECASE)

_SECTIONS = {
    "education": re.compile(r"^\s*education\b", re.IGNORECASE | re.MULTILINE),
    "experience": re.compile(r"^\s*(?:work |professional )?experience\b|^\s*employment\b", re.IGNORECASE | re.MULTILINE),
    "projects": re.compile(r"^\s*(?:\w+ )?projects\b", re.IGNORECASE | re.MULTILINE),
    "skills": re.compile(r"^\s*(?:technical )?skills\b", re.IGNORECASE | re.MULTILINE),
}

_MAX_BULLET_WORDS = 35


def lint_resume(resume_text: str) -> dict:
    bullets = split_resume_into_bullets(resume_text)
    issues: list[dict] = []

    unquantified = 0
    for bullet in bullets:
        if _WEAK_OPENERS.search(bullet):
            issues.append({"type": "weak_opener", "severity": "warning", "text": bullet,
                           "suggestion": "Start with a strong action verb (Built, Designed, Led, Reduced…)."})
        if _FIRST_PERSON.search(bullet):
            issues.append({"type": "first_person", "severity": "info", "text": bullet,
                           "suggestion": "Drop first-person pronouns — resume bullets are implied first person."})
        words = len(bullet.split())
        if words > _MAX_BULLET_WORDS:
            issues.append({"type": "too_long", "severity": "info", "text": bullet,
                           "suggestion": f"{words} words — aim for under {_MAX_BULLET_WORDS} so it fits on 1–2 lines."})
        if not _HAS_METRIC.search(bullet):
            unquantified += 1

    quantified_pct = round((len(bullets) - unquantified) / len(bullets) * 100, 1) if bullets else 0.0
    if bullets and quantified_pct < 40:
        issues.append({"type": "few_metrics", "severity": "warning", "text": "",
                       "suggestion": f"Only {quantified_pct:.0f}% of bullets include a number. Add real scale or results "
                                     "where you have them (users, latency, accuracy, data size, time saved)."})

    if not _EMAIL.search(resume_text):
        issues.append({"type": "missing_contact", "severity": "warning", "text": "",
                       "suggestion": "No email address found — make sure contact info is in plain text, not an image or header."})
    if not _LINK.search(resume_text):
        issues.append({"type": "missing_links", "severity": "info", "text": "",
                       "suggestion": "Consider adding a GitHub or LinkedIn link — especially valuable for internship applications."})

    sections_found = [name for name, pattern in _SECTIONS.items() if pattern.search(resume_text)]
    missing_sections = [s for s in ("education", "skills") if s not in sections_found]
    if missing_sections:
        issues.append({"type": "missing_sections", "severity": "info", "text": "",
                       "suggestion": f"No clear section heading for: {', '.join(missing_sections)}. "
                                     "Standard headings help ATS parsers place your content."})

    return {
        "bullet_count": len(bullets),
        "quantified_pct": quantified_pct,
        "sections_found": sections_found,
        "has_email": bool(_EMAIL.search(resume_text)),
        "has_phone": bool(_PHONE.search(resume_text)),
        "issues": issues,
    }
