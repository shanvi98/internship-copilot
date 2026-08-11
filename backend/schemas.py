"""
Pydantic schemas for validating LLM tool outputs.

Ollama (and even Claude, occasionally) can return JSON that's syntactically
valid but doesn't match the shape a prompt asked for — a missing key, a string
where a list was expected, extra prose that survived the fence-stripping.
json.loads() alone won't catch any of that; it'll happily hand back a dict
that's silently wrong, and the bug shows up two or three functions downstream
as a confusing KeyError or None.

Validating right where the LLM response is parsed turns that into one clear,
immediately-logged failure — which the orchestrator already knows how to
handle gracefully (see agent/orchestrator.py's try/except around each LLM
step, which degrades to an offline fallback rather than crashing).
"""

from pydantic import BaseModel, Field, field_validator


class JDExtractionResult(BaseModel):
    """Expected shape of extract_requirements_llm()'s output."""

    role_title: str = ""
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    seniority: str = ""
    key_responsibilities: list[str] = Field(default_factory=list)

    @field_validator("key_responsibilities")
    @classmethod
    def cap_responsibilities(cls, v: list[str]) -> list[str]:
        # Prompt asks for max 5, but don't trust the LLM to actually honor that.
        return v[:5]


class BulletRewriteResult(BaseModel):
    """Expected shape of rewrite_bullet()'s output."""

    rewritten_bullet: str
    keywords_incorporated: list[str] = Field(default_factory=list)
    confidence_note: str = ""

    @field_validator("rewritten_bullet")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("rewritten_bullet must not be empty")
        return v.strip()
