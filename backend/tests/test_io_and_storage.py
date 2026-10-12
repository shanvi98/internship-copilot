"""Tests for file text extraction, JD URL fetching (no network), and the SQLite tracker."""

import io
import socket

import pytest

from storage.tracker import Tracker
from tools import jd_fetcher
from tools.document_loader import DocumentError, extract_text
from tools.jd_fetcher import FetchError, parse_job_page


# --- document loader ---

def test_extract_plain_text_and_markdown():
    assert extract_text("resume.txt", "Line one\r\n\n\n\nLine  two".encode()) == "Line one\n\nLine two"
    assert extract_text("jd.MD", "# Title".encode()) == "# Title"


def test_extract_docx_including_tables():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Built a REST API with Flask")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skills"
    table.rows[0].cells[1].text = "Python, SQL"
    buf = io.BytesIO()
    document.save(buf)

    text = extract_text("resume.docx", buf.getvalue())
    assert "Built a REST API with Flask" in text
    assert "Skills | Python, SQL" in text


@pytest.mark.parametrize("name, data, message", [
    ("resume.exe", b"x", "Unsupported"),
    ("resume.txt", b"   \n  ", "No text"),
    ("resume.docx", b"not a zip", "Couldn't read"),
])
def test_extract_text_errors(name, data, message):
    with pytest.raises(DocumentError, match=message):
        extract_text(name, data)


# --- JD fetcher ---

def test_parse_job_page_prefers_json_ld():
    page = """<html><head><title>Careers</title>
    <script type="application/ld+json">{"@context": "https://schema.org", "@graph": [{"@type": "Organization"},
      {"@type": "JobPosting", "title": "Data Intern", "hiringOrganization": {"name": "Acme"},
       "description": "&lt;p&gt;Work with &lt;b&gt;Python&lt;/b&gt;&lt;/p&gt;&lt;ul&gt;&lt;li&gt;SQL&lt;/li&gt;&lt;/ul&gt;"}]}</script>
    </head><body><nav>Home Jobs</nav><p>Unrelated</p></body></html>"""
    result = parse_job_page(page)
    assert result["source"] == "json-ld"
    assert result["title"] == "Data Intern"
    assert result["company"] == "Acme"
    assert "Python" in result["text"] and "• SQL" in result["text"]
    assert "Unrelated" not in result["text"]


def test_parse_job_page_falls_back_to_visible_text():
    page = "<html><head><title>ML Intern</title><style>p{}</style></head><body><nav>Menu</nav>" \
           "<h1>ML Intern</h1><p>Use PyTorch daily.</p><script>var x = 1;</script><footer>(c)</footer></body></html>"
    result = parse_job_page(page)
    assert result["source"] == "page-text"
    assert result["title"] == "ML Intern"
    assert "Use PyTorch daily." in result["text"]
    assert "Menu" not in result["text"] and "var x" not in result["text"] and "(c)" not in result["text"]


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/admin",
    "http://localhost/",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/",
    "file:///etc/passwd",
    "ftp://example.com/jd.txt",
])
def test_fetch_rejects_non_public_urls(url):
    with pytest.raises(FetchError):
        jd_fetcher.fetch_job_description(url)


def test_fetch_rejects_hosts_resolving_to_private_ips(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("192.168.1.10", 80))])
    with pytest.raises(FetchError, match="private"):
        jd_fetcher.fetch_job_description("http://jobs.example.com/123")


# --- tracker ---

@pytest.fixture
def tracker(tmp_path):
    return Tracker(tmp_path / "nested" / "copilot.db")


def test_tracker_crud(tracker):
    item = tracker.create(company="Acme", role="ML Intern", match_score=72,
                          missing_skills=["docker"], analysis={"match_score": {"score": 72}})
    assert item["id"] == 1
    assert item["status"] == "saved"
    assert item["missing_skills"] == ["docker"]
    assert item["analysis"] == {"match_score": {"score": 72}}

    updated = tracker.update(item["id"], status="applied", notes="Referred by Sam")
    assert updated["status"] == "applied"
    assert updated["applied_on"]  # set automatically on first move to "applied"
    assert updated["notes"] == "Referred by Sam"

    assert [i["company"] for i in tracker.list(status="applied")] == ["Acme"]
    assert tracker.list(query="sam")[0]["id"] == item["id"]
    assert "analysis" not in tracker.list()[0]  # list view stays light

    assert tracker.delete(item["id"]) is True
    assert tracker.get(item["id"]) is None
    assert tracker.delete(item["id"]) is False
    assert tracker.update(999, notes="x") is None


def test_tracker_rejects_unknown_status(tracker):
    with pytest.raises(ValueError):
        tracker.create(company="Acme", status="ghosted")


def test_tracker_stats(tracker):
    tracker.create(company="A", status="applied", match_score=60, missing_skills=["docker", "aws"])
    tracker.create(company="B", status="interviewing", match_score=80, missing_skills=["docker"])
    tracker.create(company="C", status="saved", missing_skills=["aws", "docker"])

    stats = tracker.stats()
    assert stats["total"] == 3
    assert stats["by_status"]["applied"] == 1
    assert stats["avg_match_score"] == 70.0
    assert stats["response_rate_pct"] == 50.0
    assert stats["top_missing_skills"][0] == {"skill": "docker", "count": 3}
