"""Shared fixtures. Nothing here touches the network, Ollama, or the embedding model."""

import pytest

SAMPLE_RESUME = """Alex Rivera — alex@example.com — github.com/alexr
EDUCATION
B.S. Computer Science, State University
PROJECTS
• Built a HybridRAG customer complaint classifier using FAISS, Ollama, and Streamlit for 2,000 users.
• Responsible for a basketball analysis platform using React, FastAPI, MediaPipe, YOLOv8, and XGBoost.
• Implemented an ETL pipeline with PySpark and Databricks following medallion architecture.
SKILLS
Python, SQL, Git
"""

SAMPLE_JD = """Machine Learning Engineering Intern - Summer 2027
About the role
You'll build retrieval features for our applied AI team.
Requirements:
- Proficiency in Python and PyTorch
- Experience with Docker and AWS
Nice to have:
- Kubernetes
Benefits: free lunch.
"""


@pytest.fixture
def sample_resume():
    return SAMPLE_RESUME


@pytest.fixture
def sample_jd():
    return SAMPLE_JD


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """Point Ollama at a closed port so an accidental real call fails fast instead of hanging."""
    import config

    monkeypatch.setattr(config.settings, "ollama_host", "http://127.0.0.1:9", raising=False)
