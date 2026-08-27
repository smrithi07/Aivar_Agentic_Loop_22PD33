"""Runtime guardrails: iteration cap, token budget, stuck-loop detection."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Guardrails:
    """Loaded directly from config.yaml → agent section."""

    max_iterations: int = 20
    max_retries_per_question: int = 3
    stuck_loop_window: int = 3
    token_budget: int = 50_000

    @classmethod
    def from_config(cls, config: dict) -> "Guardrails":
        agent_cfg = config.get("agent", {})
        return cls(
            max_iterations=int(agent_cfg.get("max_iterations", 20)),
            max_retries_per_question=int(agent_cfg.get("max_retries_per_question", 3)),
            stuck_loop_window=int(agent_cfg.get("stuck_loop_window", 3)),
            token_budget=int(agent_cfg.get("token_budget", 50_000)),
        )


def check_token_budget(tokens_used: int, budget: int) -> bool:
    """Return True (halt) if token budget is exceeded."""
    if tokens_used >= budget:
        logger.warning(
            "Token budget exceeded — halting loop",
            extra={"tokens_used": tokens_used, "budget": budget},
        )
        return True
    return False


def check_stuck_loop(consecutive_no_progress: int, window: int) -> bool:
    """
    Return True (halt) if no new requirement has been covered for *window*
    consecutive iterations — indicating the agent is stuck.
    """
    if consecutive_no_progress >= window:
        logger.warning(
            "Stuck-loop detected — no progress for %d consecutive iterations",
            consecutive_no_progress,
            extra={"stuck_loop_window": window},
        )
        return True
    return False


def check_iteration_limit(iteration: int, max_iterations: int) -> bool:
    """Return True (halt) if the hard iteration cap is reached."""
    if iteration >= max_iterations:
        logger.warning(
            "Hard iteration limit reached — terminating gracefully",
            extra={"iteration": iteration, "max_iterations": max_iterations},
        )
        return True
    return False


def check_all_covered(covered: set, all_requirements: list[str]) -> bool:
    """Return True (done) if every known requirement has been addressed."""
    if all_requirements and covered >= set(all_requirements):
        logger.info(
            "All requirements covered — loop complete",
            extra={"n_questions": len(covered)},
        )
        return True
    return False
