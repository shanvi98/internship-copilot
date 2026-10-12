"""
Shared LLM access for every tool that needs one.

Both backends return JSON validated against a Pydantic schema:
  - ollama:    local and free. Uses Ollama's structured-output `format` (a JSON
               schema), falling back to plain `format: "json"` on older Ollama
               versions that don't accept a schema.
  - anthropic: paid API. Uses structured outputs (`output_config.format`), so the
               response is guaranteed to be schema-shaped JSON, and opts into
               server-side refusal fallbacks on models that support them.

Errors are split into two kinds so the orchestrator can react correctly:
  - LLMUnavailable: the backend can't be reached or isn't configured (Ollama not
    running, model not pulled, bad API key). Retrying is pointless, so the
    orchestrator stops calling the LLM for the rest of the run.
  - ValueError / LLMError: this particular response was bad (invalid JSON, schema
    mismatch, truncated). Worth a retry.
"""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from config import settings

T = TypeVar("T", bound=BaseModel)

# Models that accept `fallbacks: "default"` (beta server-side-fallback-2026-07-01).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5", "claude-sonnet-5-5"}


class LLMError(RuntimeError):
    """A single LLM call failed in a way that might succeed on retry."""


class LLMUnavailable(LLMError):
    """The LLM backend can't be used at all right now — don't retry."""


class LLMRefusal(LLMError):
    """The model declined the request."""


def _extract_json_object(text: str) -> str:
    """Strip code fences / stray prose and return the outermost {...} block."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_json_response(text: str, schema: type[T]) -> T:
    try:
        parsed = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON was not an object")
    try:
        return schema(**parsed)
    except ValidationError as e:
        # Re-raised as ValueError so callers (and the orchestrator's retry/
        # degrade logic) don't need to know or catch pydantic specifically.
        raise ValueError(f"LLM JSON didn't match expected schema: {e}") from e


def _call_ollama(prompt: str, json_schema: dict, model: str | None) -> str:
    import requests

    model = model or settings.ollama_model
    url = f"{settings.ollama_host}/api/generate"

    def post(fmt):
        return requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False, "format": fmt, "options": {"temperature": 0.2}},
            timeout=settings.llm_timeout_seconds,
        )

    try:
        resp = post(json_schema)
        if resp.status_code == 400:
            # Ollama < 0.5 only understands format="json"
            resp = post("json")
    except requests.ConnectionError as e:
        raise LLMUnavailable(f"Ollama is not reachable at {settings.ollama_host} (is `ollama serve` running?)") from e
    except requests.Timeout as e:
        raise LLMError(f"Ollama timed out after {settings.llm_timeout_seconds:.0f}s") from e

    if resp.status_code == 404:
        raise LLMUnavailable(f"Ollama model '{model}' is not available — run `ollama pull {model}`")
    try:
        resp.raise_for_status()
        return resp.json()["response"]
    except (requests.HTTPError, KeyError, ValueError) as e:
        raise LLMError(f"Ollama returned an unexpected response: {e}") from e


def make_anthropic_client():
    """Build an Anthropic client, or return None if the SDK/credentials aren't available."""
    try:
        import anthropic
    except ImportError:
        return None
    try:
        return anthropic.Anthropic(timeout=max(settings.llm_timeout_seconds, 120.0))
    except anthropic.AnthropicError:
        return None


def _call_anthropic(prompt: str, json_schema: dict, model: str | None, client) -> str:
    import anthropic

    if client is None:
        raise LLMUnavailable("No Anthropic client configured (set ANTHROPIC_API_KEY on the server)")
    model = model or settings.anthropic_model

    kwargs = {}
    if model in _FALLBACK_MODELS:
        # Re-runs a declined request on Anthropic's recommended fallback model server-side.
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            output_config={
                "effort": settings.anthropic_effort,
                "format": {"type": "json_schema", "schema": json_schema},
            },
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
        raise LLMUnavailable(f"Anthropic API rejected the request: {e.message}") from e
    except anthropic.BadRequestError as e:
        raise LLMUnavailable(f"Anthropic API bad request: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable("Could not connect to the Anthropic API") from e
    except anthropic.APIStatusError as e:
        # 429 / 5xx — the SDK already retried with backoff; let the orchestrator decide.
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e

    if response.stop_reason == "refusal":
        raise LLMRefusal("The model declined this request")
    if response.stop_reason == "max_tokens":
        raise LLMError("Anthropic response was truncated (max_tokens)")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise LLMError("Anthropic response contained no text block")
    return text


def complete_json(
    prompt: str,
    schema: type[T],
    json_schema: dict,
    backend: str = "ollama",
    model: str | None = None,
    client=None,
) -> T:
    """Send prompt to the chosen backend and return the response validated as `schema`."""
    if backend == "ollama":
        text = _call_ollama(prompt, json_schema, model)
    elif backend == "anthropic":
        text = _call_anthropic(prompt, json_schema, model, client)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return parse_json_response(text, schema)


def ollama_status(timeout: float = 1.5) -> dict:
    """Cheap reachability probe for the /capabilities endpoint."""
    import requests

    try:
        resp = requests.get(f"{settings.ollama_host}/api/tags", timeout=timeout)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        return {"reachable": False, "model": settings.ollama_model, "model_pulled": False, "models": []}
    pulled = any(m == settings.ollama_model or m.split(":")[0] == settings.ollama_model for m in models)
    return {"reachable": True, "model": settings.ollama_model, "model_pulled": pulled, "models": models}
