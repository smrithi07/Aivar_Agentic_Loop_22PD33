import json
import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.state import AgentState
    from llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def plan(
    state: "AgentState",
    gemini_client: "GeminiClient",
) -> Optional[dict]:
    """
    Plan the interview.
    Analyses the JD and resume to determine how many questions to ask and which requirements to target.
    """
    logger.info("Executing planning phase (Iteration 0)")
    
    template_path = os.path.join(_PROMPT_DIR, "planning.txt")
    with open(template_path, encoding="utf-8") as fh:
        template = fh.read()

    reqs_summary = json.dumps(state.requirements, indent=2)

    prompt = template.format(
        structured_requirements=reqs_summary,
        resume_text=state.resume_text,
    )

    try:
        raw = gemini_client.generate(prompt)
        parsed = gemini_client._parse_json(raw)
        if not parsed or "plan" not in parsed:
            raise ValueError("Planning LLM response missing 'plan' array")
            
        interview_plan = parsed["plan"]
        logger.info(
            "Interview plan created",
            extra={
                "is_mismatch": parsed.get("is_mismatch", False),
                "target_question_count": parsed.get("target_question_count", len(interview_plan)),
                "planned_questions": len(interview_plan),
            }
        )
        return parsed
    except Exception as exc:
        logger.error("Planning phase failed: %s", exc)
        return None
