"""Fallback handlers for each failure mode in InterviewLens."""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM / generation fallbacks
# ---------------------------------------------------------------------------

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Try to extract a JSON object from an LLM response that may be wrapped
    in markdown code fences or preceded by prose.

    Returns the parsed dict, or None if extraction fails.
    """
    if not text:
        return None

    # 1) Try markdown fenced block  ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        candidate = match.group(1).strip()
    else:
        # 2) Try to find a raw JSON object {}
        match = re.search(r"(\{[\s\S]*\})", text)
        candidate = match.group(1).strip() if match else text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning(
            "JSON extraction failed",
            extra={"raw_text_snippet": text[:200]},
        )
        return None


def handle_malformed_generation(
    raw_response: str,
    target_requirement: str,
    tool_result: Optional[dict] = None,
) -> dict:
    """
    Fallback when Gemini returns an unparseable generation response.
    Returns a minimal valid question dict so the loop can continue.
    """
    logger.warning(
        "Malformed generation response — using fallback question",
        extra={"target_requirement": target_requirement},
    )
    evidence = (tool_result or {}).get("evidence", [])
    excerpt = str(evidence[0].get("excerpt", "")).strip() if evidence else ""
    if excerpt:
        question = (
            f"Your resume mentions: '{excerpt}' How did you approach this work, "
            "what trade-offs did you make, and what measurable outcome did it produce?"
        )
        resume_connection = excerpt
        relevance = (
            f"Connects the resume evidence to the job requirement: {target_requirement}."
        )
        answer_hint = (
            "Explain your role, the design decisions and constraints, and quantify "
            "the result. Relate the experience back to the role's requirements."
        )
    else:
        question = f"Can you describe your experience with {target_requirement}?"
        resume_connection = None
        relevance = f"Directly tests the required skill: {target_requirement}"
        answer_hint = (
            "Walk through a specific project or role where you applied this skill, "
            "focusing on the problem, your approach, and the outcome."
        )

    return {
        "question": question,
        "category": "Technical",
        "target_skill": target_requirement,
        "relevance": relevance,
        "resume_connection": resume_connection,
        "answer_hint": answer_hint,
        "_fallback": True,
    }


def handle_malformed_reflection(raw_response: str) -> dict:
    """
    Fallback when the reflection Gemini call returns unparseable output.
    Defaults to 'accept' to avoid infinite retry loops on reflection failures.
    """
    logger.warning(
        "Malformed reflection response — defaulting to accept",
        extra={"raw_snippet": raw_response[:200]},
    )
    return {
        "verdict": "accept",
        "feedback": "Reflection parsing failed; accepted by fallback.",
        "scores": {
            "jd_relevance": 3,
            "resume_grounding": 3,
            "clarity": 3,
            "usefulness": 3,
            "target_skill_alignment": 3,
            "originality": 3,
        },
        "overall_score": 3.0,
        "_fallback": True,
    }


def handle_malformed_reasoning(raw_response: str, uncovered: list[str]) -> Optional[dict]:
    """
    Fallback when reasoning returns unparseable output.
    Picks the first uncovered requirement and defaults to jd_based.
    """
    if not uncovered:
        return None
    req = uncovered[0]
    logger.warning(
        "Malformed reasoning response — falling back to first uncovered requirement",
        extra={"chosen_requirement": req},
    )
    return {
        "thought": "Fallback: picking first uncovered requirement.",
        "action": "none",
        "observation": "No resume evidence checked (fallback).",
        "target_requirement": req,
        "question_type": "jd_based",
        "tool_to_call": "none",
        "tool_args": {},
        "reasoning_trace": "Fallback reasoning used due to parse failure.",
    }


# ---------------------------------------------------------------------------
# Tool failure fallbacks
# ---------------------------------------------------------------------------

def handle_extract_requirements_failure(jd_text: str) -> dict:
    """
    Fallback when extract_requirements tool fails entirely.
    Extracts bullet-point lines that look like real skill requirements,
    then sanitizes them with the same filter used in the happy path.
    """
    # Import here to avoid circular imports
    from tools.extract_requirements import _sanitize_skills  # noqa: PLC0415

    logger.warning("extract_requirements failed — using raw JD as single requirement block")

    lines = [ln.strip().lstrip("-•*").strip() for ln in jd_text.splitlines() if ln.strip()]

    # Prefer lines that started with a bullet (likely real requirements)
    bullet_lines = [
        ln.strip().lstrip("-•*").strip()
        for ln in jd_text.splitlines()
        if ln.strip() and ln.strip()[0] in "-•*"
    ]
    candidates = bullet_lines if bullet_lines else lines

    # Sanitize with the same filter used in the happy path
    skills = _sanitize_skills([ln for ln in candidates if len(ln) > 5])[:10]

    return {
        "required_skills": skills[:5] if skills else ["General engineering skills"],
        "preferred_skills": [],
        "responsibilities": skills[5:] if len(skills) > 5 else [],
        "seniority": "Unknown",
    }


def handle_find_resume_evidence_failure(requirement: str) -> dict:
    """
    Fallback when find_resume_evidence tool fails.
    Returns a no-evidence result so the agent generates a JD-based question.
    """
    logger.warning(
        "find_resume_evidence failed — returning empty evidence",
        extra={"requirement": requirement},
    )
    return {
        "evidence": [],
        "has_evidence": False,
    }


# ---------------------------------------------------------------------------
# Memory failure fallbacks
# ---------------------------------------------------------------------------

def handle_memory_recall_failure(query: str) -> list:
    """
    Fallback when memory.recall() raises an exception.
    Returns an empty list so the loop can continue without memory context.
    """
    logger.warning(
        "memory.recall() failed — proceeding with empty context",
        extra={"query": query},
    )
    return []


def handle_memory_save_failure(text: str, metadata: dict) -> None:
    """
    Fallback when memory.save() raises an exception.
    Memory failure is non-fatal: log and continue.
    """
    logger.warning(
        "memory.save() failed — entry will not be persisted",
        extra={"text_snippet": text[:100], "metadata_type": metadata.get("type")},
    )
