"""
Tests for tools/llm.py and tools/bullet_writer.py with the LLM mocked out.

The point of these is the contract the orchestrator relies on: an unreachable
backend raises LLMUnavailable (don't retry), a bad response raises ValueError
(retry), and a rewrite that invents facts never reaches the user.
"""

from types import SimpleNamespace

import pytest
import requests

from schemas import BulletRewriteResult, JDExtractionResult
from tools import bullet_writer, llm
from tools.bullet_writer import UnsupportedClaimError, rewrite_bullet, verify_rewrite


# --- JSON parsing ---

def test_parse_json_strips_fences_and_prose():
    text = 'Sure! Here you go:\n```json\n{"rewritten_bullet": "Built X", "keywords_incorporated": []}\n```'
    result = llm.parse_json_response(text, BulletRewriteResult)
    assert result.rewritten_bullet == "Built X"


def test_parse_json_rejects_wrong_shape():
    with pytest.raises(ValueError, match="schema"):
        llm.parse_json_response('{"keywords_incorporated": []}', BulletRewriteResult)
    with pytest.raises(ValueError, match="valid JSON"):
        llm.parse_json_response("not json at all", BulletRewriteResult)


def test_jd_schema_dedupes_skills_and_caps_responsibilities():
    result = JDExtractionResult(
        must_have_skills=["Python", "python ", "", "SQL"],
        key_responsibilities=[str(i) for i in range(9)],
    )
    assert result.must_have_skills == ["Python", "SQL"]
    assert len(result.key_responsibilities) == 5


# --- Ollama backend ---

def test_ollama_connection_error_is_unavailable(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(llm.LLMUnavailable, match="not reachable"):
        llm.complete_json("prompt", BulletRewriteResult, {}, backend="ollama")


def test_ollama_missing_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(status_code=404))
    with pytest.raises(llm.LLMUnavailable, match="ollama pull"):
        llm.complete_json("prompt", BulletRewriteResult, {}, backend="ollama")


def test_ollama_falls_back_to_plain_json_format(monkeypatch):
    calls = []

    class Resp:
        def __init__(self, status):
            self.status_code = status

        def raise_for_status(self):
            pass

        def json(self):
            return {"response": '{"rewritten_bullet": "Built Y"}'}

    def fake_post(url, json, timeout):
        calls.append(json["format"])
        return Resp(400 if isinstance(json["format"], dict) else 200)

    monkeypatch.setattr(requests, "post", fake_post)
    result = llm.complete_json("prompt", BulletRewriteResult, {"type": "object"}, backend="ollama")
    assert result.rewritten_bullet == "Built Y"
    assert calls == [{"type": "object"}, "json"]


# --- Anthropic backend (fake client, no network) ---

class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _fake_client(stop_reason="end_turn", text='{"rewritten_bullet": "Built Z"}'):
    response = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])
    messages = _FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_anthropic_uses_structured_output_and_fallbacks():
    client, messages = _fake_client()
    result = llm.complete_json("prompt", BulletRewriteResult, {"type": "object"}, backend="anthropic", client=client)

    assert result.rewritten_bullet == "Built Z"
    assert messages.kwargs["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}
    assert messages.kwargs["fallbacks"] == "default"
    assert messages.kwargs["betas"] == ["server-side-fallback-2026-07-01"]


def test_anthropic_refusal_raises():
    client, _ = _fake_client(stop_reason="refusal", text="")
    with pytest.raises(llm.LLMRefusal):
        llm.complete_json("prompt", BulletRewriteResult, {}, backend="anthropic", client=client)


def test_anthropic_without_client_is_unavailable():
    with pytest.raises(llm.LLMUnavailable):
        llm.complete_json("prompt", BulletRewriteResult, {}, backend="anthropic", client=None)


# --- Truthfulness guard ---

def test_verify_rewrite_accepts_faithful_rephrasing():
    original = "Built a resume parser using NLP techniques and scikit-learn"
    rewritten = "Developed an NLP-based machine learning resume parser with scikit-learn"
    assert verify_rewrite(original, rewritten) == []


def test_verify_rewrite_flags_invented_numbers_and_tools():
    original = "Built a REST API with Flask"
    problems = verify_rewrite(original, "Built a REST API with Flask and Docker serving 10,000 users")
    assert any("10000" in p for p in problems)
    assert any("docker" in p for p in problems)


def test_rewrite_bullet_discards_invented_claims(monkeypatch):
    monkeypatch.setattr(bullet_writer, "complete_json", lambda *a, **k: BulletRewriteResult(
        rewritten_bullet="Built a Flask REST API on AWS improving latency by 40%",
        keywords_incorporated=["AWS"],
    ))
    with pytest.raises(UnsupportedClaimError):
        rewrite_bullet("Built a REST API with Flask", ["aws"])


def test_rewrite_bullet_only_reports_keywords_actually_used(monkeypatch):
    monkeypatch.setattr(bullet_writer, "complete_json", lambda *a, **k: BulletRewriteResult(
        rewritten_bullet="Developed RESTful API endpoints with Flask",
        keywords_incorporated=["RESTful", "Kubernetes"],
    ))
    result = rewrite_bullet("Built a REST API with Flask", ["rest api", "kubernetes"])
    assert result["keywords_incorporated"] == ["RESTful"]
