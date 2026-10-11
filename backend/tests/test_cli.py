"""Tests for the command-line interface (offline --quick mode only)."""

import json

import pytest

import cli
import config


@pytest.fixture
def files(tmp_path, sample_resume, sample_jd):
    resume = tmp_path / "resume.txt"
    jd = tmp_path / "jd.md"
    resume.write_text(sample_resume)
    jd.write_text(sample_jd)
    return resume, jd


def test_cli_quick_markdown_report(files, capsys):
    resume, jd = files
    assert cli.main(["--resume", str(resume), "--jd", str(jd), "--quick"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Match report — Machine Learning Engineering Intern")
    assert "**Match score:" in out
    assert "[high] **pytorch**" in out


def test_cli_quick_json_and_save(files, capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "db_path", tmp_path / "cli.db")
    resume, jd = files
    cli.main(["--resume", str(resume), "--jd", str(jd), "--quick", "--json", "--save", "--company", "Acme"])
    captured = capsys.readouterr()
    assert json.loads(captured.out)["gap_analysis"]["missing_skills"]
    assert "Saved to tracker as #1" in captured.err

    from storage.tracker import Tracker
    item = Tracker(tmp_path / "cli.db").get(1)
    assert item["company"] == "Acme"
    assert item["role"] == "Machine Learning Engineering Intern - Summer 2027"


def test_cli_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit, match="file not found"):
        cli.main(["--resume", str(tmp_path / "nope.txt"), "--jd", str(tmp_path / "nope.md"), "--quick"])
