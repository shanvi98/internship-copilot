"""
Tool 1: JD Parser
Extracts structured requirements (skills, tools, seniority, keywords) from a raw
job description. Two paths:
  - extract_skills_offline(): fast taxonomy match, no API call, always works.
    Also detects whether each skill sits under a "required" or a
    "preferred / nice to have" part of the JD, so gaps can be prioritized
    even when no LLM is available.
  - extract_requirements_llm(): richer structured extraction via an LLM, used by the agent
"""

import re
from dataclasses import dataclass, field

from schemas import JDExtractionResult
from tools.llm import complete_json
from tools.skill_taxonomy import SKILL_NAMES, find_skill_spans

# Kept for backwards compatibility with code/tests that import the flat list.
SKILL_TAXONOMY = SKILL_NAMES

REQUIRED = "required"
PREFERRED = "preferred"
MENTIONED = "mentioned"

# Section headings that switch the "current requirement level" for following lines.
_PREFERRED_HEADING = re.compile(
    r"\b(preferred|nice[- ]to[- ]haves?|bonus(?: points)?|pluses|desired|good to have|"
    r"would be great|extra credit|ideal(?:ly)?)\b",
    re.IGNORECASE,
)
_REQUIRED_HEADING = re.compile(
    r"\b(requirements?|required|qualifications|must[- ]haves?|what you(?:'|’)?ll need|"
    r"what we(?:'|’)?re looking for|you have|you bring|who you are|skills)\b",
    re.IGNORECASE,
)
# Neutral headings end a requirements section (e.g. "Benefits", "About us").
_NEUTRAL_HEADING = re.compile(
    r"\b(about (?:us|the (?:team|company|role))|benefits|perks|compensation|salary|"
    r"responsibilities|what you(?:'|’)?ll do|the role|our team|equal opportunity)\b",
    re.IGNORECASE,
)
# Inline cues that override the section for a single line/sentence.
_PREFERRED_CUE = re.compile(
    r"\b(a plus|is a plus|are a plus|plus\b|preferred|nice to have|bonus|familiarity with|"
    r"exposure to|ideally|not required|optional)",
    re.IGNORECASE,
)
_REQUIRED_CUE = re.compile(
    r"\b(required|requires?|must|need(?:s|ed)? to|proficien(?:t|cy)|strong|solid|"
    r"experience (?:with|in|using)|knowledge of)\b",
    re.IGNORECASE,
)

_SENIORITY_PATTERNS = [
    ("intern", r"\b(?:intern|internship|co-?op)s?\b"),
    ("new grad", r"\bnew[- ]grad(?:uate)?s?\b"),
    ("entry level", r"\bentry[- ]level\b"),
    ("junior", r"\bjunior\b|\bjr\.?\b"),
    ("mid level", r"\bmid[- ]level\b"),
    ("senior", r"\bsenior\b|\bsr\.?\b"),
    ("staff", r"\bstaff (?:engineer|scientist|software|ml|data)\b"),
    ("principal", r"\bprincipal\b"),
]

_ROLE_WORDS = r"(?:intern(?:ship)?|engineer|scientist|developer|analyst|researcher|architect|designer|manager)"
_ROLE_LINE = re.compile(rf"(?im)^\s*([^\n]{{3,90}}?\b{_ROLE_WORDS}\b[^\n]{{0,60}})\s*$")


@dataclass
class JDRequirements:
    raw_text: str
    matched_skills: list[str] = field(default_factory=list)
    role_title: str | None = None
    seniority_hint: str | None = None
    # skill -> "required" | "preferred" | "mentioned"
    requirement_levels: dict[str, str] = field(default_factory=dict)


def _is_heading(line: str) -> bool:
    stripped = line.strip().strip("#*_ ").strip()
    if not stripped:
        return False
    return stripped.endswith(":") or (len(stripped.split()) <= 6 and not stripped.endswith("."))


def _line_levels(text: str) -> list[tuple[int, int, str]]:
    """
    Split text into lines and assign each a requirement level, based on the most
    recent section heading plus any inline cue on the line itself.
    Returns [(start_offset, end_offset, level), ...].
    """
    out = []
    section = MENTIONED
    offset = 0
    for line in text.splitlines(keepends=True):
        start, end = offset, offset + len(line)
        offset = end
        content = line.strip()
        if not content:
            continue

        if _is_heading(content):
            if _PREFERRED_HEADING.search(content):
                section = PREFERRED
            elif _NEUTRAL_HEADING.search(content):
                section = MENTIONED
            elif _REQUIRED_HEADING.search(content):
                section = REQUIRED

        # Inline cues are checked per sentence, so "Python required. Go is a plus."
        # classifies the two skills differently.
        sent_offset = start
        for sentence in re.split(r"(?<=[.;!?])\s+", line):
            s_start = text.find(sentence, sent_offset) if sentence else sent_offset
            s_end = s_start + len(sentence)
            sent_offset = s_end
            if _PREFERRED_CUE.search(sentence):
                level = PREFERRED
            elif section == PREFERRED:
                level = PREFERRED
            elif section == REQUIRED or _REQUIRED_CUE.search(sentence):
                level = REQUIRED
            else:
                level = MENTIONED
            out.append((s_start, s_end, level))
    return out


_LEVEL_RANK = {REQUIRED: 3, MENTIONED: 2, PREFERRED: 1}


def detect_requirement_levels(text: str, spans: dict[str, list[tuple[int, int]]] | None = None) -> dict[str, str]:
    """For each skill in text: the strongest requirement level it appears under."""
    spans = spans if spans is not None else find_skill_spans(text)
    regions = _line_levels(text)
    levels: dict[str, str] = {}
    for skill, positions in spans.items():
        best = None
        for pos, _ in positions:
            level = next((lvl for s, e, lvl in regions if s <= pos < e), MENTIONED)
            if best is None or _LEVEL_RANK[level] > _LEVEL_RANK[best]:
                best = level
        levels[skill] = best or MENTIONED
    return levels


def detect_seniority(text: str) -> str | None:
    for label, pattern in _SENIORITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return None


def detect_role_title(text: str) -> str | None:
    match = _ROLE_LINE.search(text)
    if not match:
        return None
    title = match.group(1).strip().strip("#*-•:| ").strip()
    return title or None


def extract_skills_offline(jd_text: str) -> JDRequirements:
    """
    Fast, deterministic skill extraction. No API call — safe to run on every
    JD as a first pass, and works even if the LLM call later fails or is rate limited.
    """
    spans = find_skill_spans(jd_text)
    return JDRequirements(
        raw_text=jd_text,
        matched_skills=sorted(spans),
        role_title=detect_role_title(jd_text),
        seniority_hint=detect_seniority(jd_text),
        requirement_levels=detect_requirement_levels(jd_text, spans),
    )


EXTRACTION_PROMPT = """You are analyzing a job/internship description. Extract ONLY what's explicitly stated.

Return JSON with this exact shape:
{{
  "role_title": string,
  "company": string (empty if not stated),
  "must_have_skills": [string],
  "nice_to_have_skills": [string],
  "seniority": string,
  "key_responsibilities": [string, max 5 items, each under 15 words]
}}

Skills should be short names of tools, languages, frameworks or concepts (e.g. "PyTorch", "SQL", "REST APIs"), not full sentences.

Job description:
{jd_text}
"""

# Strict JSON schema for backends that support constrained decoding.
JD_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "role_title": {"type": "string"},
        "company": {"type": "string"},
        "must_have_skills": {"type": "array", "items": {"type": "string"}},
        "nice_to_have_skills": {"type": "array", "items": {"type": "string"}},
        "seniority": {"type": "string"},
        "key_responsibilities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["role_title", "company", "must_have_skills", "nice_to_have_skills", "seniority", "key_responsibilities"],
    "additionalProperties": False,
}


def extract_requirements_llm(jd_text: str, backend: str = "ollama", model: str | None = None, client=None) -> dict:
    """
    Richer structured extraction using an LLM. Pluggable backend:
      - backend="ollama" (default, FREE, runs locally): needs `ollama serve` running
        and a model pulled, e.g. `ollama pull llama3.1:8b`
      - backend="anthropic": needs an Anthropic client + ANTHROPIC_API_KEY, paid

    Falls back to extract_skills_offline() should be handled by the caller if this raises.
    """
    prompt = EXTRACTION_PROMPT.format(jd_text=jd_text)
    result = complete_json(
        prompt, JDExtractionResult, JD_EXTRACTION_SCHEMA, backend=backend, model=model, client=client
    )
    return result.model_dump()


if __name__ == "__main__":
    # quick offline test — no API key needed
    sample_jd = """
    Machine Learning Engineering Intern - Summer 2027

    We're looking for a junior ML intern to join our applied AI team.
    You'll work with PyTorch, build RAG pipelines using LangChain and FAISS,
    and deploy models via FastAPI and Docker on AWS. Experience with
    scikit-learn and SQL is a plus. Familiarity with LangGraph or agent
    frameworks is a strong bonus.
    """
    result = extract_skills_offline(sample_jd)
    print("Role title:", result.role_title)
    print("Seniority:", result.seniority_hint)
    print("Matched skills:", result.matched_skills)
    print("Requirement levels:", result.requirement_levels)
