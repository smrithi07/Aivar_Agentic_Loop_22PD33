"""
AgentState — single source of truth for the running agent.

Passed by reference through every phase function.  All mutations to
iteration progress, token usage, and question history happen here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentState:
    """Holds all mutable state for a single InterviewLens run."""

    # ---- Inputs (immutable after initialisation) ----
    jd_text: str = ""
    resume_text: str = ""

    # ---- Structured requirements (populated after extract_requirements) ----
    requirements: dict = field(default_factory=dict)

    # ---- Interview Plan (populated after planning phase) ----
    interview_plan: list[dict] = field(default_factory=list)

    # ---- Coverage tracking ----
    covered_requirements: set = field(default_factory=set)

    # ---- Accepted questions ----
    questions: list[dict] = field(default_factory=list)

    # ---- Loop counters ----
    iteration: int = 0
    consecutive_no_progress: int = 0
    retry_count: int = 0  # retries for the current question

    # ---- Resource tracking ----
    tokens_used: int = 0

    def all_requirement_strings(self) -> list[str]:
        """
        Flatten the structured requirements into a single list of strings.
        Used by guardrails.check_all_covered() and reason().
        """
        reqs: list[str] = []
        reqs.extend(self.requirements.get("required_skills", []))
        reqs.extend(self.requirements.get("preferred_skills", []))
        reqs.extend(self.requirements.get("responsibilities", []))
        return reqs

    def uncovered_requirements(self) -> list[str]:
        """Return requirement strings not yet in covered_requirements."""
        return [r for r in self.all_requirement_strings() if r not in self.covered_requirements]

    def sync_tokens(self, gemini_client) -> None:
        """Sync token usage from the GeminiClient into state."""
        self.tokens_used = gemini_client.total_tokens_used
