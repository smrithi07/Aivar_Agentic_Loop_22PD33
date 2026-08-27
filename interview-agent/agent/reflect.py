"""
REFLECT phase — Reflexion quality-control pattern.

Evaluates every generated question against a rubric before it is accepted
into the question set.  On failure, returns a "retry" verdict with specific
feedback so that act() can regenerate an improved question.

On acceptance, the reflection result is saved to semantic memory so that
future reason() calls can recall quality signals and avoid similar failures.

Memory flow: reflect() → save()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from harness.fallback import handle_malformed_reflection
from harness.retry import with_retry

if TYPE_CHECKING:
    from agent.act import ActResult
    from agent.reason import ReasoningDecision
    from agent.state import AgentState
    from llm.gemini_client import GeminiClient
    from memory.memory import SemanticMemory

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


@dataclass
class ReflectionResult:
    """Output of the REFLECT phase."""

    verdict: str        # "accept" | "retry"
    feedback: str
    scores: dict
    overall_score: float
    _fallback: bool = False


def reflect(
    act_result: "ActResult",
    decision: "ReasoningDecision",
    state: "AgentState",
    memory: "SemanticMemory",
    gemini_client: "GeminiClient",
    config: dict,
) -> ReflectionResult:
    """
    Evaluate *act_result* using the Reflexion rubric.

    Saves the reflection result to memory on both accept AND retry so that
    future iterations can recall quality signals.
    """
    # ── Step 1: build reflection prompt ──────────────────────────────────
    prompt = _build_reflection_prompt(act_result, decision, state)

    # ── Step 2: call Gemini ───────────────────────────────────────────────
    try:
        raw = gemini_client.generate(prompt)
        parsed = gemini_client._parse_json(raw)
        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            else:
                raise ValueError("Expected a dict, got a list")
        elif parsed is not None and not isinstance(parsed, dict):
            raise ValueError(f"Expected a dict, got {type(parsed)}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("reflect() generation failed — accepting with fallback: %s", exc)
        raw = ""
        parsed = None

    if parsed is None:
        fallback = handle_malformed_reflection(raw)
        result = ReflectionResult(
            verdict=fallback["verdict"],
            feedback=fallback["feedback"],
            scores=fallback["scores"],
            overall_score=fallback["overall_score"],
            _fallback=True,
        )
    else:
        verdict = str(parsed.get("verdict", "retry"))
        if verdict not in ("accept", "retry"):
            verdict = "retry"

        scores = parsed.get("scores", {})
        overall = float(parsed.get("overall_score", _compute_avg(scores)))

        # Force retry if any individual score is below 3
        if any(v < 3 for v in scores.values() if isinstance(v, (int, float))):
            if verdict == "accept":
                verdict = "retry"
                feedback = (
                    str(parsed.get("feedback", "")) +
                    " [Auto-retry: at least one rubric score is below threshold.]"
                )
            else:
                feedback = str(parsed.get("feedback", ""))
        else:
            feedback = str(parsed.get("feedback", ""))

        result = ReflectionResult(
            verdict=verdict,
            feedback=feedback,
            scores=scores,
            overall_score=overall,
        )

    # ── Step 3: save to memory (always — both accept and retry) ──────────
    _save_reflection_to_memory(act_result, decision, result, state.iteration, memory)

    logger.info(
        "reflect() verdict",
        extra={
            "verdict": result.verdict,
            "overall_score": result.overall_score,
            "target_requirement": decision.target_requirement,
            "iteration": state.iteration,
            "is_fallback": result._fallback,
        },
    )
    return result


@dataclass
class BatchReflectionResult:
    accepted_indices: list[int]
    discarded_feedback: list[dict]
    missing_areas: list[str]


def batch_reflect(
    act_results: list["ActResult"],
    state: "AgentState",
    memory: "SemanticMemory",
    gemini_client: "GeminiClient",
) -> BatchReflectionResult:
    """
    Evaluate an entire batch of questions at once.
    """
    prompt = _build_batch_reflection_prompt(act_results, state)

    def _call_and_parse():
        raw_resp = gemini_client.generate(prompt)
        parsed_json = gemini_client._parse_json(raw_resp)
        if not parsed_json:
            raise ValueError("Parsed JSON is None")
        if isinstance(parsed_json, list):
            if len(parsed_json) == 1 and isinstance(parsed_json[0], dict):
                parsed_json = parsed_json[0]
            else:
                raise ValueError("Expected a dict, got a list")
        elif not isinstance(parsed_json, dict):
            raise ValueError(f"Expected a dict, got {type(parsed_json)}")
        return raw_resp, parsed_json

    try:
        raw, parsed = with_retry(_call_and_parse, gemini_client.retry_config)
            
        result = BatchReflectionResult(
            accepted_indices=parsed.get("accepted_indices", []),
            discarded_feedback=parsed.get("discarded_feedback", []),
            missing_areas=parsed.get("missing_areas", []),
        )

        # Save to memory so Iteration 1+ can recall quality signals
        for i, act_result in enumerate(act_results):
            if i in result.accepted_indices:
                text = (
                    f"Batch Reflection on '{act_result.target_requirement}': "
                    f"verdict=accept. The question was deemed high quality."
                )
                memory.save(text=text, metadata={
                    "type": "batch_reflection",
                    "target_requirement": act_result.target_requirement,
                    "verdict": "accept"
                })
            else:
                feedback = next((d.get("reason", "Unknown") for d in result.discarded_feedback if d.get("index") == i), "Failed rubric")
                text = (
                    f"Batch Reflection on '{act_result.target_requirement}': "
                    f"verdict=discard. Feedback: {feedback[:200]}"
                )
                memory.save(text=text, metadata={
                    "type": "batch_reflection",
                    "target_requirement": act_result.target_requirement,
                    "verdict": "discard",
                    "feedback": feedback
                })

        return result
    except Exception as exc:
        logger.warning("batch_reflect() generation failed: %s", exc)
        # Fallback: accept all, no missing areas
        return BatchReflectionResult(
            accepted_indices=list(range(len(act_results))),
            discarded_feedback=[],
            missing_areas=[],
        )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_reflection_prompt(act_result: "ActResult", decision: "ReasoningDecision", state: "AgentState") -> str:
    template_path = os.path.join(_PROMPT_DIR, "reflection.txt")
    try:
        with open(template_path, encoding="utf-8") as fh:
            template = fh.read()
    except FileNotFoundError:
        template = _DEFAULT_REFLECTION_TEMPLATE

    question_json = json.dumps(act_result.to_dict(), indent=2)
    existing_q_list = (
        "\n".join(f"  - {q.get('question', '')[:120]}" for q in state.questions)
        or "  (none yet)"
    )

    return template.format(
        question_json=question_json,
        existing_questions=existing_q_list,
        target_requirement=decision.target_requirement,
        question_type=act_result.question_type,
    )


def _build_batch_reflection_prompt(act_results: list["ActResult"], state: "AgentState") -> str:
    template_path = os.path.join(_PROMPT_DIR, "batch_reflection.txt")
    with open(template_path, encoding="utf-8") as fh:
        template = fh.read()

    questions_json = json.dumps([q.to_dict() for q in act_results], indent=2)
    reqs_str = json.dumps(state.requirements, indent=2)
    plan_str = json.dumps(state.interview_plan, indent=2)

    return template.format(
        questions_json=questions_json,
        requirements=reqs_str,
        interview_plan=plan_str,
    )


def _save_reflection_to_memory(
    act_result: "ActResult",
    decision: "ReasoningDecision",
    result: "ReflectionResult",
    iteration: int,
    memory: "SemanticMemory",
) -> None:
    """Persist the reflection verdict and quality signal to semantic memory."""
    text = (
        f"Reflection on question about '{decision.target_requirement}': "
        f"verdict={result.verdict}, score={result.overall_score:.1f}. "
        f"Feedback: {result.feedback[:200]}"
    )
    metadata = {
        "type": "reflection_feedback",
        "iteration": iteration,
        "target_requirement": decision.target_requirement,
        "question_type": decision.question_type,
        "verdict": result.verdict,
        "overall_score": result.overall_score,
        "scores": result.scores,
    }
    memory.save(text=text, metadata=metadata)


def _compute_avg(scores: dict) -> float:
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Fallback template
# ---------------------------------------------------------------------------

_DEFAULT_REFLECTION_TEMPLATE = """
You are a quality reviewer for interview questions. Evaluate the question below.

## Question to Evaluate
{question_json}

## Question Type
{question_type}

## Existing Questions (check for duplication)
{existing_questions}

## Target JD Requirement
{target_requirement}

## Rubric (score each 1–5, where 5 = excellent)

1. jd_relevance: Does the question address a real JD requirement?
2. resume_grounding: For combined/resume_based — is the resume connection accurate? For jd_based — score 5.
3. clarity: Is the question unambiguous and professionally phrased?
4. usefulness: Would this reveal meaningful signal about the candidate?
5. target_skill_alignment: Does it clearly test the stated target_skill?
6. originality: Is it meaningfully different from existing questions?

## Minimum Threshold
All scores >= 3. Average >= 3.5. If ANY criterion fails, verdict must be "retry".

Return ONLY a JSON object:
{{
  "verdict": "accept | retry",
  "feedback": "Specific explanation. If retry, say exactly what to fix.",
  "scores": {{
    "jd_relevance": 0,
    "resume_grounding": 0,
    "clarity": 0,
    "usefulness": 0,
    "target_skill_alignment": 0,
    "originality": 0
  }},
  "overall_score": 0.0
}}
""".strip()
