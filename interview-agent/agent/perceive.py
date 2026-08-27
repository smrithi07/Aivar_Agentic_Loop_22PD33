"""
PERCEIVE phase — build a PerceptionContext from all available sources.

The perception context aggregates:
  - Raw JD and resume texts
  - Structured requirements (already extracted)
  - Uncovered requirements (delta between all requirements and covered set)
  - Generated questions so far (for de-duplication awareness)
  - Memory context retrieved via recall() — always called before returning

Memory recall happens here so that every downstream component (reason, act,
reflect) has access to prior context without needing to call memory directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.state import AgentState
    from memory.memory import SemanticMemory

logger = logging.getLogger(__name__)


@dataclass
class PerceptionContext:
    """Snapshot of everything the agent knows at the start of an iteration."""

    raw_jd: str
    raw_resume: str
    structured_requirements: dict
    uncovered_requirements: list[str]
    generated_questions: list[dict]
    memory_context: list[dict]      # results of memory.recall()
    iteration: int
    tokens_used: int


def perceive(state: "AgentState", memory: "SemanticMemory", config: dict) -> PerceptionContext:
    """
    Build a PerceptionContext for the current iteration.

    Calls memory.recall() with a generic coverage query so that earlier
    reflection feedback and covered-requirement entries are surfaced.
    """
    uncovered = state.uncovered_requirements()

    # Build a query that surfaces both coverage state and prior quality notes
    recall_query = (
        f"covered requirements and reflection feedback after iteration {state.iteration}"
        if state.iteration > 0
        else "job description requirements and candidate experience"
    )

    memory_context = memory.recall(recall_query, top_k=config.get("memory", {}).get("top_k", 3))

    logger.debug(
        "perceive complete",
        extra={
            "iteration": state.iteration,
            "uncovered_count": len(uncovered),
            "memory_context_count": len(memory_context),
        },
    )

    return PerceptionContext(
        raw_jd=state.jd_text,
        raw_resume=state.resume_text,
        structured_requirements=state.requirements,
        uncovered_requirements=uncovered,
        generated_questions=state.questions,
        memory_context=memory_context,
        iteration=state.iteration,
        tokens_used=state.tokens_used,
    )
