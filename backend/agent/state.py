"""
Shared state passed through the agent pipeline. Every step reads from and
writes to this object, and every step logs what it did (or failed to do) —
this is what lets us show, transparently, what the agent decided at each stage
(useful for the README/demo, and for debugging when a step degrades).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepLog:
    step_name: str
    status: str  # "ok" | "degraded" | "failed"
    detail: str = ""


@dataclass
class AgentState:
    resume_text: str
    jd_text: str

    # populated as the pipeline runs
    jd_requirements: dict[str, Any] | None = None
    jd_skills_offline: list[str] = field(default_factory=list)
    resume_match: dict[str, Any] | None = None
    gap_analysis: dict[str, Any] | None = None
    prioritized_gaps: list[dict] = field(default_factory=list)
    rewritten_bullets: list[dict] = field(default_factory=list)

    logs: list[StepLog] = field(default_factory=list)

    def log(self, step_name: str, status: str, detail: str = ""):
        self.logs.append(StepLog(step_name=step_name, status=status, detail=detail))

    def summary(self) -> dict:
        return {
            "steps": [{"step": l.step_name, "status": l.status, "detail": l.detail} for l in self.logs],
            "jd_requirements": self.jd_requirements,
            "resume_match": self.resume_match,
            "gap_analysis": self.gap_analysis,
            "prioritized_gaps": self.prioritized_gaps,
            "rewritten_bullets": self.rewritten_bullets,
        }