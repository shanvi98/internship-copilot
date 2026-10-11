"""
End-to-end tests of the FastAPI app through TestClient. The LLM is either
unreachable (the default from conftest) or monkeypatched; embeddings are
disabled per request, so these run in well under a second.
"""

import pytest
from fastapi.testclient import TestClient

import main
from agent import orchestrator


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "db_path", tmp_path / "test.db")
    monkeypatch.setattr(orchestrator.time, "sleep", lambda s: None)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_root_redirects_to_ui(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/ui/"


def test_capabilities_reports_unreachable_ollama(client):
    caps = client.get("/capabilities").json()
    assert caps["ollama"]["reachable"] is False
    assert caps["anthropic"]["configured"] in (True, False)


def test_quick_analysis_keeps_v01_fields(client, sample_resume, sample_jd):
    r = client.post("/analyze/quick", json={"resume_text": sample_resume, "jd_text": sample_jd})
    assert r.status_code == 200
    body = r.json()
    for key in ("matched_skills", "missing_skills", "bonus_skills", "coverage_pct", "prioritized_gaps",
                "gap_analysis", "keyword_analysis", "resume_checks", "match_score"):
        assert key in body
    assert "pytorch" in body["missing_skills"]


def test_analyze_degrades_when_llm_down(client, sample_resume, sample_jd):
    r = client.post("/analyze", json={"resume_text": sample_resume, "jd_text": sample_jd, "use_embeddings": False})
    assert r.status_code == 200
    steps = {s["step"]: s["status"] for s in r.json()["steps"]}
    assert steps["jd_extraction"] == "degraded"
    assert steps["bullet_rewriting"] == "skipped"


@pytest.mark.parametrize("payload, status", [
    ({"resume_text": "short", "jd_text": "also short"}, 422),
    ({"resume_text": " " * 30 + "x", "jd_text": "x" * 30}, 422),
    ({"resume_text": "x" * 30, "jd_text": "y" * 30, "llm_backend": "gpt"}, 422),
    ({"resume_text": "x" * 30, "jd_text": "y" * 30, "max_rewrites": 50}, 422),
])
def test_analyze_validation(client, payload, status):
    assert client.post("/analyze", json=payload).status_code == status


def test_anthropic_without_key_is_400(client, monkeypatch):
    monkeypatch.setattr(main, "_anthropic_client", None)
    r = client.post("/analyze", json={"resume_text": "x" * 30, "jd_text": "y" * 30, "llm_backend": "anthropic"})
    assert r.status_code == 400
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_rewrite_endpoint(client, monkeypatch):
    monkeypatch.setattr(main, "rewrite_bullet", lambda bullet, skills, **k: {
        "rewritten_bullet": "Developed RESTful API endpoints with Flask",
        "keywords_incorporated": ["RESTful"], "confidence_note": "", "_skills": skills,
    })
    r = client.post("/rewrite", json={"bullet": "Built a REST API with Flask", "jd_text": "We use Flask and Docker."})
    assert r.status_code == 200
    assert r.json()["original"] == "Built a REST API with Flask"
    assert set(r.json()["_skills"]) == {"flask", "docker"}

    assert client.post("/rewrite", json={"bullet": "Built a REST API with Flask"}).status_code == 400


def test_rewrite_endpoint_llm_down_is_503(client):
    r = client.post("/rewrite", json={"bullet": "Built a REST API with Flask", "jd_skills": ["docker"]})
    assert r.status_code == 503


def test_extract_text_upload(client):
    r = client.post("/extract-text", files={"file": ("resume.txt", b"Built a REST API with Flask", "text/plain")})
    assert r.status_code == 200
    assert r.json()["text"] == "Built a REST API with Flask"

    r = client.post("/extract-text", files={"file": ("resume.png", b"\x89PNG", "image/png")})
    assert r.status_code == 400


def test_jd_fetch_blocks_internal_urls(client):
    r = client.post("/jd/fetch", json={"url": "http://127.0.0.1:8000/health"})
    assert r.status_code == 400


def test_jd_fetch_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(main.settings, "allow_url_fetch", False)
    assert client.post("/jd/fetch", json={"url": "https://example.com/job"}).status_code == 403


def test_applications_crud(client):
    r = client.post("/applications", json={"company": "Acme", "role": "ML Intern", "match_score": 64,
                                           "missing_skills": ["docker"], "analysis": {"match_score": {"score": 64}}})
    assert r.status_code == 201
    app_id = r.json()["id"]

    assert client.get(f"/applications/{app_id}").json()["analysis"] == {"match_score": {"score": 64}}
    assert client.patch(f"/applications/{app_id}", json={"status": "interviewing"}).json()["status"] == "interviewing"
    assert client.patch(f"/applications/{app_id}", json={"status": "ghosted"}).status_code == 422
    assert [a["id"] for a in client.get("/applications", params={"status": "interviewing"}).json()] == [app_id]
    assert client.get("/applications/stats").json()["by_status"]["interviewing"] == 1

    assert client.delete(f"/applications/{app_id}").status_code == 204
    assert client.get(f"/applications/{app_id}").status_code == 404
    assert client.delete(f"/applications/{app_id}").status_code == 404


def test_ui_is_served(client):
    r = client.get("/ui/")
    assert r.status_code == 200
    assert "Internship Application Copilot" in r.text
