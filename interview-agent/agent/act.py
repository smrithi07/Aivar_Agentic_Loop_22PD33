"""
ACT phase — execute the selected tool and generate the interview question.

Flow:
  1. Run the tool chosen by reason() (find_resume_evidence or none).
  2. Build a generation prompt with requirement + tool result + optional retry feedback.
  3. Call Gemini to produce a structured question payload.
  4. Return an ActResult dataclass.

Falls back gracefully on tool failure and on malformed Gemini output.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from harness.fallback import (
    handle_find_resume_evidence_failure,
    handle_malformed_generation,
)
from harness.retry import with_retry
from tools.find_resume_evidence import find_resume_evidence

if TYPE_CHECKING:
    from agent.reason import ReasoningDecision
    from agent.state import AgentState
    from llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


@dataclass
class ActResult:
    """Output of the ACT phase — a fully annotated interview question."""

    question: str
    category: str
    target_skill: str
    relevance: str
    resume_connection: Optional[str]
    answer_hint: str
    question_type: str
    target_requirement: str
    tool_result: dict
    raw_response: str = ""
    _fallback: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "category": self.category,
            "target_skill": self.target_skill,
            "relevance": self.relevance,
            "resume_connection": self.resume_connection,
            "answer_hint": self.answer_hint,
            "question_type": self.question_type,
            "target_requirement": self.target_requirement,
        }


def act(
    decision: "ReasoningDecision",
    state: "AgentState",
    gemini_client: "GeminiClient",
    config: dict,
    retry_feedback: Optional[str] = None,
) -> ActResult:
    """
    Execute the tool selected by reason() and generate the question.

    Parameters
    ----------
    decision : ReasoningDecision
    state : AgentState
    gemini_client : GeminiClient
    config : dict
    retry_feedback : str, optional
        If this is a retry attempt, the reflection feedback from the previous try.
    """
    tool_result: dict = {}

    # ── Step 1: execute tool ──────────────────────────────────────────────
    if decision.tool_to_call == "find_resume_evidence":
        try:
            tool_result = find_resume_evidence(
                requirement=decision.tool_args.get("requirement", decision.target_requirement),
                resume_text=decision.tool_args.get("resume_text", state.resume_text),
                gemini_client=gemini_client,
                top_k=int(config.get("tools", {}).get("find_resume_evidence_top_k", 3)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool find_resume_evidence raised: %s", exc)
            tool_result = handle_find_resume_evidence_failure(decision.target_requirement)

        # If evidence is missing, downgrade combined/resume_based to jd_based
        if not tool_result.get("has_evidence", False):
            if decision.question_type in ("combined", "resume_based"):
                logger.info(
                    "No resume evidence found — downgrading question_type to jd_based",
                    extra={"requirement": decision.target_requirement},
                )
                decision.question_type = "jd_based"

    # ── Step 2: build generation prompt ──────────────────────────────────
    resume_evidence_str = _format_evidence(tool_result)
    prompt = _build_generation_prompt(
        decision=decision,
        resume_evidence=resume_evidence_str,
        retry_feedback=retry_feedback or "",
    )

    # ── Step 3: call Gemini ───────────────────────────────────────────────
    try:
        raw = gemini_client.generate(prompt)
        parsed = gemini_client._parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("act() generation failed — using local fallback: %s", exc)
        raw = ""
        parsed = None

    if parsed is None:
        fallback_q = handle_malformed_generation(
            raw, decision.target_requirement, tool_result=tool_result
        )
        return ActResult(
            question=fallback_q["question"],
            category=fallback_q["category"],
            target_skill=fallback_q["target_skill"],
            relevance=fallback_q["relevance"],
            resume_connection=fallback_q.get("resume_connection"),
            answer_hint=fallback_q["answer_hint"],
            question_type=decision.question_type,
            target_requirement=decision.target_requirement,
            tool_result=tool_result,
            raw_response=raw,
            _fallback=True,
        )

    # ── Step 4: build ActResult ───────────────────────────────────────────
    result = ActResult(
        question=str(parsed.get("question", "")).strip(),
        category=str(parsed.get("category", "Technical")),
        target_skill=str(parsed.get("target_skill", decision.target_requirement)),
        relevance=str(parsed.get("relevance", "")),
        resume_connection=parsed.get("resume_connection") or None,
        answer_hint=str(parsed.get("answer_hint", "")),
        question_type=decision.question_type,
        target_requirement=decision.target_requirement,
        tool_result=tool_result,
        raw_response=raw,
    )

    logger.info(
        "act() question generated",
        extra={
            "question_type": result.question_type,
            "target_skill": result.target_skill,
            "is_fallback": result._fallback,
            "is_retry": retry_feedback is not None,
        },
    )
    return result


def batch_act(
    interview_plan: list[dict],
    state: "AgentState",
    gemini_client: "GeminiClient",
) -> list[ActResult]:
    """
    Generate all planned questions in a single LLM call.
    """
    prompt = _build_batch_generation_prompt(
        interview_plan=interview_plan,
        resume_snippet=state.resume_text[:1500],
    )
    
    def _call_and_parse():
        raw_resp = gemini_client.generate(prompt)
        parsed = gemini_client._parse_json(raw_resp)
        if not isinstance(parsed, list):
            raise ValueError("Batch generation did not return a JSON array")
        return raw_resp, parsed

    try:
        raw, parsed_array = with_retry(_call_and_parse, gemini_client.retry_config)
    except Exception as exc:
        logger.warning("batch_act() generation failed: %s", exc)
        return []

    results = []
    for i, parsed in enumerate(parsed_array):
        # We try to correlate the generated question with the plan using the index, 
        # assuming the LLM generated them in order.
        plan_item = interview_plan[i] if i < len(interview_plan) else {}
        
        target_requirement = str(plan_item.get("target_requirement", "General"))
        question_type = str(plan_item.get("question_type", "jd_based"))
        
        result = ActResult(
            question=str(parsed.get("question", "")).strip(),
            category=str(parsed.get("category", "Technical")),
            target_skill=str(parsed.get("target_skill", target_requirement)),
            relevance=str(parsed.get("relevance", "")),
            resume_connection=parsed.get("resume_connection") or None,
            answer_hint=str(parsed.get("answer_hint", "")),
            question_type=question_type,
            target_requirement=target_requirement,
            tool_result={},
            raw_response=raw,
        )
        results.append(result)

    logger.info("batch_act() generated %d questions", len(results))
    return results


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_generation_prompt(
    decision: "ReasoningDecision",
    resume_evidence: str,
    retry_feedback: str,
) -> str:
    template_path = os.path.join(_PROMPT_DIR, "generation.txt")
    try:
        with open(template_path, encoding="utf-8") as fh:
            template = fh.read()
    except FileNotFoundError:
        template = _DEFAULT_GENERATION_TEMPLATE

    return template.format(
        target_requirement=decision.target_requirement,
        question_type=decision.question_type,
        resume_evidence=resume_evidence,
        retry_feedback=retry_feedback or "None — this is the first attempt.",
        reasoning_trace=decision.reasoning_trace,
    )


def _build_batch_generation_prompt(
    interview_plan: list[dict],
    resume_snippet: str,
) -> str:
    template_path = os.path.join(_PROMPT_DIR, "batch_generation.txt")
    with open(template_path, encoding="utf-8") as fh:
        template = fh.read()

    plan_json = json.dumps(interview_plan, indent=2)

    return template.format(
        interview_plan_json=plan_json,
        resume_snippet=resume_snippet,
    )


def _format_evidence(tool_result: dict) -> str:
    if not tool_result or not tool_result.get("has_evidence", False):
        return "No resume evidence found for this requirement."
    lines = []
    for ev in tool_result.get("evidence", []):
        section = ev.get("section", "General")
        excerpt = ev.get("excerpt", "")
        score = ev.get("relevance_score", 0)
        lines.append(f"  [{section}] (score: {score:.2f}) {excerpt}")
    return "\n".join(lines) if lines else "No evidence items."


# ---------------------------------------------------------------------------
# Fallback template (used if prompts/generation.txt is missing)
# ---------------------------------------------------------------------------

_DEFAULT_GENERATION_TEMPLATE = """
You are an expert technical interviewer. Generate ONE high-quality interview question.

Target Requirement: {target_requirement}
Question Type: {question_type}
Reasoning Context: {reasoning_trace}

Resume Evidence:
{resume_evidence}

Retry Feedback (incorporate if provided):
{retry_feedback}

Question Types:
- jd_based: derived entirely from the job requirement, no resume reference needed.
- resume_based: probes a specific item from the candidate's resume.
- combined: explicitly connects the JD requirement with specific resume evidence.

Return ONLY a JSON object:
{{
  "question": "The interview question (open-ended, not yes/no)",
  "category": "Technical | Behavioral | Situational | Role-Fit",
  "target_skill": "The specific skill being assessed",
  "relevance": "Why this question matters for the role",
  "resume_connection": "Verbatim or close-paraphrase resume excerpt, or null",
  "answer_hint": "How the candidate can connect their genuine experience to the answer without inventing experience"
}}
""".strip()
