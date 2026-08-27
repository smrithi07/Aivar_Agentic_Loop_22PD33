"""
REASON phase — ReAct reasoning pattern.

Implements Thought → Action → Observation → Decision using a Gemini prompt
that explicitly models the ReAct loop.

Key responsibilities:
  1. Call memory.recall(target_requirement) to surface prior context BEFORE
     making any decision.
  2. Build a prompt with full context (requirements, uncovered list, memory).
  3. Call Gemini to produce a structured reasoning decision.
  4. Return a ReasoningDecision dataclass.

The recall() call is the mechanism by which memory from an earlier iteration
changes behaviour in a later one: if a requirement was saved as
"covered_requirement" in memory, the model's reasoning prompt will include
that fact and it will choose a different requirement.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from harness.fallback import handle_malformed_reasoning

if TYPE_CHECKING:
    from agent.perceive import PerceptionContext
    from llm.gemini_client import GeminiClient
    from memory.memory import SemanticMemory

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


@dataclass
class ReasoningDecision:
    """Output of the REASON phase."""

    target_requirement: str
    question_type: str          # "jd_based" | "resume_based" | "combined"
    tool_to_call: str           # "find_resume_evidence" | "none"
    tool_args: dict
    reasoning_trace: str        # full Thought-Action-Observation for logging


def reason(
    perception: "PerceptionContext",
    memory: "SemanticMemory",
    gemini_client: "GeminiClient",
    config: dict,
) -> Optional[ReasoningDecision]:
    """
    Determine which requirement to address next and how.

    Returns None if there are no uncovered requirements left.
    """
    if not perception.uncovered_requirements:
        logger.info("reason(): no uncovered requirements — signalling loop completion")
        return None

    # ── Step 1: recall from memory before reasoning ──────────────────────
    # Query with the first uncovered requirement as an anchor to surface
    # any relevant prior coverage or quality feedback.
    recall_query = perception.uncovered_requirements[0]
    prior_context = memory.recall(recall_query)

    # ── Step 2: build the reasoning prompt ───────────────────────────────
    prompt = _build_reasoning_prompt(perception, prior_context)

    # ── Step 3: call Gemini ───────────────────────────────────────────────
    raw = ""  # ensure `raw` is always defined even if generate() raises
    try:
        raw = gemini_client.generate(prompt)
        parsed = gemini_client._parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reason() generation failed — using local fallback: %s", exc)
        parsed = handle_malformed_reasoning("", perception.uncovered_requirements)

    if parsed is None:
        fallback = handle_malformed_reasoning(raw, perception.uncovered_requirements)
        if fallback is None:
            return None
        parsed = fallback

    # ── Step 4: validate and build decision ──────────────────────────────
    target = str(parsed.get("target_requirement", "")).strip()
    if not target:
        target = perception.uncovered_requirements[0]

    question_type = str(parsed.get("question_type", "jd_based"))
    if question_type not in ("jd_based", "resume_based", "combined"):
        question_type = "jd_based"

    tool_to_call = str(parsed.get("tool_to_call", "none"))
    if tool_to_call not in ("find_resume_evidence", "none"):
        tool_to_call = "none"

    tool_args = parsed.get("tool_args", {})
    if tool_to_call == "find_resume_evidence" and not tool_args:
        tool_args = {
            "requirement": target,
            "resume_text": perception.raw_resume,
        }

    reasoning_trace = str(parsed.get("reasoning_trace", ""))

    decision = ReasoningDecision(
        target_requirement=target,
        question_type=question_type,
        tool_to_call=tool_to_call,
        tool_args=tool_args,
        reasoning_trace=reasoning_trace,
    )

    logger.info(
        "reason() decision",
        extra={
            "target_requirement": target,
            "question_type": question_type,
            "tool_to_call": tool_to_call,
            "iteration": perception.iteration,
        },
    )
    return decision


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_reasoning_prompt(
    perception: "PerceptionContext",
    prior_memory: list[dict],
) -> str:
    """Construct the full ReAct reasoning prompt."""

    template_path = os.path.join(_PROMPT_DIR, "reasoning.txt")
    try:
        with open(template_path, encoding="utf-8") as fh:
            template = fh.read()
    except FileNotFoundError:
        template = _DEFAULT_REASONING_TEMPLATE

    reqs_summary = json.dumps(perception.structured_requirements, indent=2)
    uncovered_list = "\n".join(
        f"  - {r}" for r in perception.uncovered_requirements
    ) or "  (none remaining)"

    covered_list = "\n".join(
        f"  - {q.get('target_skill', '?')}"
        for q in perception.generated_questions
    ) or "  (none yet)"

    memory_str = _format_memory(prior_memory)

    return template.format(
        uncovered_requirements=uncovered_list,
        covered_requirements=covered_list,
        memory_context=memory_str,
        iteration=perception.iteration,
        resume_snippet=perception.raw_resume[:500],
    )


def _format_memory(memory_entries: list[dict]) -> str:
    if not memory_entries:
        return "  (no prior memory entries)"
    lines = []
    for entry in memory_entries:
        entry_type = entry.get("type", "unknown")
        text = entry.get("text", "")[:200]
        lines.append(f"  [{entry_type}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fallback prompt (used if prompts/reasoning.txt is missing)
# ---------------------------------------------------------------------------

_DEFAULT_REASONING_TEMPLATE = """
You are an expert interview question strategist using ReAct reasoning.

## Uncovered Requirements (select ONE from this list)
{uncovered_requirements}

## Already Covered (DO NOT select these again)
{covered_requirements}

## Memory Context (from previous iterations — treat as ground truth)
{memory_context}

## Resume Snippet (first 500 chars)
{resume_snippet}

## Your Task (Iteration {iteration})

Apply ReAct: Thought → Action → Observation → Decision.

IMPORTANT:
- Select a requirement from the "Uncovered Requirements" list ONLY.
- If memory shows a requirement was already covered, it will NOT appear in the uncovered list — ignore it.
- If the resume suggests relevant experience for the selected requirement, prefer "combined" or "resume_based".
- If no resume evidence is evident, use "jd_based".

Return ONLY a JSON object:
{{
  "thought": "...",
  "action": "find_resume_evidence | none",
  "observation": "...",
  "target_requirement": "exact requirement string from the uncovered list",
  "question_type": "jd_based | resume_based | combined",
  "tool_to_call": "find_resume_evidence | none",
  "tool_args": {{"requirement": "...", "resume_text": "..."}} or {{}},
  "reasoning_trace": "brief Thought-Action-Observation summary"
}}
""".strip()
