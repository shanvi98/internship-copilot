"""
Tool 1: JD Parser
Extracts structured requirements (skills, tools, seniority, keywords) from a raw
job description. Two paths:
  - extract_skills_offline(): fast regex/taxonomy match, no API call, always works
  - extract_requirements_llm(): richer structured extraction via Claude, used by the agent
"""

import re
import json
import os  # noqa: F401 (used by extract_requirements_llm for OLLAMA_HOST)
from dataclasses import dataclass, field

# A living taxonomy of skills/tools relevant to SWE/ML internships.
# This is intentionally a plain list so it's easy to extend later.
SKILL_TAXONOMY = [
    "python", "java", "c++", "javascript", "typescript", "sql", "r",
    "react", "node.js", "fastapi", "flask", "django", "spring boot",
    "pytorch", "tensorflow", "scikit-learn", "xgboost", "keras",
    "langchain", "langgraph", "crewai", "huggingface", "transformers",
    "faiss", "pinecone", "vector database", "rag", "llm", "openai api",
    "mediapipe", "yolo", "opencv", "computer vision", "nlp",
    "docker", "kubernetes", "aws", "gcp", "azure", "ci/cd",
    "git", "github actions", "airflow", "spark", "pyspark", "databricks",
    "delta lake", "etl", "data pipeline",
    "rest api", "graphql", "microservices", "distributed systems",
    "machine learning", "deep learning", "statistics", "linear algebra",
    "a/b testing", "mlops", "model deployment", "model monitoring",
    "agile", "scrum",
]


@dataclass
class JDRequirements:
    raw_text: str
    matched_skills: list[str] = field(default_factory=list)
    role_title: str | None = None
    seniority_hint: str | None = None


def extract_skills_offline(jd_text: str) -> JDRequirements:
    """
    Fast, deterministic skill extraction. No API call — safe to run on every
    JD as a first pass, and works even if the LLM call later fails or is rate limited.
    """
    text_lower = jd_text.lower()
    matched = []
    for skill in SKILL_TAXONOMY:
        # word-boundary-ish match so "r" doesn't match inside "developer"
        pattern = r"(?<![a-zA-Z])" + re.escape(skill) + r"(?![a-zA-Z])"
        if re.search(pattern, text_lower):
            matched.append(skill)

    seniority = None
    for hint in ["intern", "new grad", "entry level", "junior", "senior", "staff"]:
        if hint in text_lower:
            seniority = hint
            break

    title_match = re.search(r"(?im)^(.*(intern|engineer|scientist|developer).*)$", jd_text)
    role_title = title_match.group(1).strip() if title_match else None

    return JDRequirements(
        raw_text=jd_text,
        matched_skills=sorted(set(matched)),
        role_title=role_title,
        seniority_hint=seniority,
    )


EXTRACTION_PROMPT = """You are analyzing a job/internship description. Extract ONLY what's explicitly stated.

Return strict JSON (no markdown fences, no preamble) with this exact shape:
{{
  "role_title": string,
  "must_have_skills": [string],
  "nice_to_have_skills": [string],
  "seniority": string,
  "key_responsibilities": [string, max 5 items, each under 15 words]
}}

Job description:
{jd_text}
"""


def extract_requirements_llm(jd_text: str, backend: str = "ollama", model: str | None = None, client=None) -> dict:
    """
    Richer structured extraction using an LLM. Pluggable backend:
      - backend="ollama" (default, FREE, runs locally): needs `ollama serve` running
        and a model pulled, e.g. `ollama pull llama3.1:8b`
      - backend="anthropic": needs an Anthropic client + ANTHROPIC_API_KEY, paid

    Falls back to extract_skills_offline() should be handled by the caller if this raises.
    """
    prompt = EXTRACTION_PROMPT.format(jd_text=jd_text)

    if backend == "ollama":
        import requests as _requests
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = model or "llama3.1:8b"
        resp = _requests.post(
            f"{ollama_host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["response"].strip()

    elif backend == "anthropic":
        if client is None:
            raise RuntimeError("No Anthropic client provided for backend='anthropic'")
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

    else:
        raise ValueError(f"Unknown backend: {backend}")

    # defensive: strip accidental code fences from either backend
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


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
