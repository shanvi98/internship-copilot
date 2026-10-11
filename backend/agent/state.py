"""
Shared state passed through the agent pipeline. Every step reads from and
writes to this object, and every step logs what it did (or failed to do) —
this is what lets us show, transparently, what the agent decided at each stage
(useful for the README/demo, and for debugging when a step degrades).
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepLog:
    step_name: str
    status: str  # "ok" | "degraded" | "failed" | "skipped"
    detail: str = ""
    duration_ms: int = 0


@dataclass
class AgentState:
    resume_text: str
    jd_text: str
    llm_backend: str = "ollama"

    # populated as the pipeline runs
    jd_requirements: dict[str, Any] | None = None
    jd_skills_offline: list[str] = field(default_factory=list)
    jd_overview: dict[str, Any] = field(default_factory=dict)  # offline title/seniority, always present
    resume_match: dict[str, Any] | None = None
    gap_analysis: dict[str, Any] | None = None
    prioritized_gaps: list[dict] = field(default_factory=list)
    keyword_analysis: dict[str, Any] | None = None
    resume_checks: dict[str, Any] | None = None
    match_score: dict[str, Any] | None = None
    rewritten_bullets: list[dict] = field(default_factory=list)

    # Set when the LLM backend is unreachable, so later steps skip it instead of
    # each paying for their own retries + timeouts.
    llm_unavailable_reason: str | None = None

    logs: list[StepLog] = field(default_factory=list)
    _step_started: float = field(default=0.0, repr=False)

    def log(self, step_name: str, status: str, detail: str = ""):
        elapsed = int((time.perf_counter() - self._step_started) * 1000) if self._step_started else 0
        self.logs.append(StepLog(step_name=step_name, status=status, detail=detail, duration_ms=elapsed))

    def start_step(self):
        """Start the clock for a step; the next log() call records its duration."""
        self._step_started = time.perf_counter()

    def summary(self) -> dict:
        return {
            "steps": [
                {"step": l.step_name, "status": l.status, "detail": l.detail, "duration_ms": l.duration_ms}
                for l in self.logs
            ],
            "llm_backend": self.llm_backend,
            "jd_overview": self.jd_overview,
            "jd_requirements": self.jd_requirements,
            "resume_match": self.resume_match,
            "gap_analysis": self.gap_analysis,
            "prioritized_gaps": self.prioritized_gaps,
            "keyword_analysis": self.keyword_analysis,
            "resume_checks": self.resume_checks,
            "match_score": self.match_score,
            "rewritten_bullets": self.rewritten_bullets,
        }
