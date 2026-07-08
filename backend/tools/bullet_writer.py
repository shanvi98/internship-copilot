"""
Tool 4: Bullet Rewriter
Takes a resume bullet + the JD's key requirements, and rewrites the bullet to
mirror the JD's language (helps with ATS keyword matching) while staying
truthful to what the person actually did — this tool must NEVER invent
metrics or claims the person didn't provide.

Default backend: Ollama (free, local). Optional: Anthropic API.
"""

import json
import os
import re

REWRITE_PROMPT = """You are helping a student tailor ONE resume bullet point to better match a job description, for ATS keyword alignment.

STRICT RULES:
- Do NOT invent numbers, metrics, or outcomes that aren't implied by the original bullet.
- Do NOT claim tools/skills the original bullet doesn't mention.
- You MAY rephrase using synonyms/terms from the JD if they accurately describe what's already there (e.g. "built a REST API" -> "developed RESTful microservices" is fine if true; adding "achieved 99.9% uptime" when not stated is NOT fine).
- Keep it one line, resume-bullet style, starting with a strong action verb.
- Return STRICT JSON only, no markdown fences, no preamble:
{{"rewritten_bullet": string, "keywords_incorporated": [string], "confidence_note": string}}

confidence_note should flag if you were unable to naturally incorporate JD keywords without overstating the original.

ORIGINAL BULLET:
{original_bullet}

JD KEY SKILLS/REQUIREMENTS:
{jd_skills}
"""


def rewrite_bullet(original_bullet: str, jd_skills: list[str], backend: str = "ollama", model: str | None = None, client=None) -> dict:
    prompt = REWRITE_PROMPT.format(
        original_bullet=original_bullet,
        jd_skills=", ".join(jd_skills),
    )

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
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

    else:
        raise ValueError(f"Unknown backend: {backend}")

    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


if __name__ == "__main__":
    # This requires `ollama serve` running locally with a model pulled.
    # We'll test this together once Ollama is confirmed running on your machine.
    print("This tool needs Ollama running locally (same setup as your HybridRAG project).")
    print("Run: ollama serve   (in one terminal)")
    print("Run: ollama pull llama3.1:8b   (one-time download, ~4.7GB)")
    print("Then re-run this file to see a live test.")
