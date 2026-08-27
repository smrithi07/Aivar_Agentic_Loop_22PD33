"""
AgentLoop — Plan-Batch-Iterate orchestration.

This is the central orchestration logic for InterviewLens.

Workflow:
1. Extract requirements
2. Iteration 0: Plan the interview -> Batch Generate Questions -> Batch Reflect
3. Iteration 1+: Reason about missing gaps -> Generate single question -> Reflect
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Generator

from agent.act import act, batch_act
from agent.perceive import perceive
from agent.plan import plan
from agent.reason import reason
from agent.reflect import reflect, batch_reflect
from agent.state import AgentState
from harness.guardrails import (
    Guardrails,
    check_all_covered,
    check_iteration_limit,
    check_stuck_loop,
    check_token_budget,
)
from harness.logger import StepTimer, log_step
from tools.extract_requirements import extract_requirements
from tools.find_resume_evidence import clear_resume_cache

if TYPE_CHECKING:
    from llm.gemini_client import GeminiClient
    from memory.memory import SemanticMemory

logger = logging.getLogger(__name__)


class AgentLoop:
    def __init__(
        self,
        gemini_client: "GeminiClient",
        memory: "SemanticMemory",
        config: dict,
    ) -> None:
        self.gemini_client = gemini_client
        self.memory = memory
        self.config = config
        self.guardrails = Guardrails.from_config(config)

    def run(self, jd: str, resume: str) -> list[dict]:
        """Backward-compatible synchronous runner for CLI."""
        questions = []
        for event in self.run_stream(jd, resume):
            if event["type"] == "done":
                questions = event["data"]
        return questions

    def run_stream(self, jd: str, resume: str) -> Generator[dict, None, None]:
        self.memory.clear()
        clear_resume_cache()
        state = AgentState(jd_text=jd, resume_text=resume)
        
        yield {"type": "start", "message": "AgentLoop started"}

        logger.info("AgentLoop started", extra={"jd_length": len(jd), "resume_length": len(resume)})

        # ── Initial requirements extraction (once per run) ────────────────
        yield {"type": "step", "action": "perceive", "message": "Extracting requirements from JD..."}
        with StepTimer() as t:
            reqs = extract_requirements(jd, self.gemini_client)
        state.requirements = reqs
        state.sync_tokens(self.gemini_client)

        self.memory.save(
            text=f"JD Requirements: {json.dumps(reqs)}",
            metadata={"type": "jd_requirements", "iteration": 0, "data": reqs},
        )

        all_reqs = state.all_requirement_strings()
        logger.info(
            "Requirements extracted",
            extra={"total_requirements": len(all_reqs), "seniority": reqs.get("seniority")},
        )

        prev_covered_count = 0

        prev_covered_count = 0
        target_question_count = 0

        # ── Main loop ─────────────────────────────────────────────────────
        for iteration in range(self.guardrails.max_iterations):
            state.iteration = iteration

            # Guardrail checks
            if check_iteration_limit(iteration, self.guardrails.max_iterations):
                break
            state.sync_tokens(self.gemini_client)
            if check_token_budget(state.tokens_used, self.guardrails.token_budget):
                break
            
            # ── Iteration 0: Plan & Batch Generate ────────────────────────
            if iteration == 0:
                yield {"type": "step", "action": "plan", "message": "Planning interview strategy based on JD and Resume..."}
                with StepTimer() as t:
                    plan_result = plan(state, self.gemini_client)
                
                if not plan_result:
                    logger.error("Planning phase failed, cannot proceed.")
                    break
                    
                if plan_result.get("is_mismatch"):
                    logger.info("Resume is a complete mismatch with JD — aborting generation.")
                    state.questions.append({
                        "question_type": "MISMATCH",
                        "category": "Fit Assessment",
                        "target_skill": "Role Fit",
                        "question": plan_result.get("mismatch_message", "Candidate resume does not match the job description."),
                        "relevance": "We want to ensure candidates have the baseline experience required before proceeding with a technical interview.",
                        "answer_hint": "Please review the job description closely and ensure your experience aligns before reapplying."
                    })
                    yield {"type": "step", "action": "plan", "message": "Resume mismatch detected. Aborting generation."}
                    break
                    
                interview_plan = plan_result.get("plan", [])
                state.interview_plan = interview_plan
                target_question_count = plan_result.get("target_question_count", len(interview_plan))
                
                yield {"type": "step", "action": "plan", "message": f"Planned {target_question_count} questions."}
                
                if not interview_plan:
                    logger.info("Plan is empty, skipping batch generation.")
                    continue
                
                yield {"type": "step", "action": "act", "message": f"Batch generating initial questions..."}
                with StepTimer() as t:
                    batch_results = batch_act(interview_plan, state, self.gemini_client)
                
                if not batch_results:
                    logger.error("Batch generation failed, cannot proceed.")
                    break
                    
                yield {"type": "step", "action": "reflect", "message": f"Reflecting on {len(batch_results)} generated questions..."}
                with StepTimer() as t:
                    reflection = batch_reflect(batch_results, state, self.memory, self.gemini_client)
                
                # Process accepted questions
                for idx in reflection.accepted_indices:
                    if idx < len(batch_results):
                        result = batch_results[idx]
                        state.questions.append(result.to_dict())
                        state.covered_requirements.add(result.target_requirement)
                        
                        self.memory.save(
                            text=(
                                f"Generated question about '{result.target_requirement}': "
                                f"{result.question}"
                            ),
                            metadata={
                                "type": "generated_question",
                                "iteration": iteration,
                                "target_requirement": result.target_requirement,
                                "question_type": result.question_type,
                                "target_skill": result.target_skill,
                            },
                        )
                        self.memory.save(
                            text=f"Covered requirement: {result.target_requirement}",
                            metadata={
                                "type": "covered_requirement",
                                "iteration": iteration,
                                "requirement": result.target_requirement,
                            },
                        )
                        
                # Save missing areas to memory so reason() can pick them up
                for missing in reflection.missing_areas:
                    self.memory.save(
                        text=f"Missing Area Identified: {missing}",
                        metadata={"type": "missing_area", "iteration": iteration}
                    )
                target_question_count += len(reflection.missing_areas)
                
                state.sync_tokens(self.gemini_client)
                prev_covered_count = len(state.covered_requirements)
                
                yield {"type": "progress", "data": state.questions}
                continue

            # ── Iteration 1+: Gap Filling ─────────────────────────────────
            
            if len(state.questions) >= target_question_count:
                logger.info("Target question count reached — stopping loop")
                break
            if check_stuck_loop(state.consecutive_no_progress, self.guardrails.stuck_loop_window):
                break

            with StepTimer() as t:
                perception = perceive(state, self.memory, self.config)
            
            if not perception.uncovered_requirements:
                logger.info("No uncovered requirements — stopping loop")
                break

            yield {"type": "step", "action": "reason", "message": f"Reasoning about missing gaps (Iteration {iteration})..."}
            with StepTimer() as t:
                decision = reason(perception, self.memory, self.gemini_client, self.config)
            state.sync_tokens(self.gemini_client)

            if decision is None:
                logger.info("reason() returned None — stopping loop")
                break

            if decision.target_requirement in state.covered_requirements:
                state.consecutive_no_progress += 1
                continue

            act_result = None
            reflect_result = None

            for retry_num in range(self.guardrails.max_retries_per_question + 1):
                retry_feedback = (
                    reflect_result.feedback if (reflect_result and reflect_result.verdict == "retry")
                    else None
                )
                
                yield {"type": "step", "action": "act", "message": f"Generating targeted question for '{decision.target_requirement}'..."}
                with StepTimer() as t:
                    try:
                        act_result = act(
                            decision, state, self.gemini_client, self.config,
                            retry_feedback=retry_feedback,
                        )
                    except Exception as exc:
                        logger.error("act() raised unexpectedly: %s", exc)
                        break

                state.sync_tokens(self.gemini_client)
                
                yield {"type": "step", "action": "reflect", "message": f"Evaluating generated question..."}
                with StepTimer() as t:
                    try:
                        reflect_result = reflect(
                            act_result, decision, state,
                            self.memory, self.gemini_client, self.config,
                        )
                    except Exception as exc:
                        logger.error("reflect() raised unexpectedly: %s", exc)
                        break

                state.sync_tokens(self.gemini_client)
                if reflect_result and reflect_result.verdict == "accept":
                    break

            reflection_exhausted = (reflect_result is not None and reflect_result.verdict == "retry")
            if act_result is not None and (
                (reflect_result is not None and reflect_result.verdict == "accept")
                or reflection_exhausted
            ):
                self.memory.save(
                    text=f"Generated question about '{decision.target_requirement}': {act_result.question}",
                    metadata={
                        "type": "generated_question",
                        "iteration": iteration,
                        "target_requirement": decision.target_requirement,
                        "question_type": decision.question_type,
                        "target_skill": act_result.target_skill,
                    },
                )
                self.memory.save(
                    text=f"Covered requirement: {decision.target_requirement}",
                    metadata={
                        "type": "covered_requirement",
                        "iteration": iteration,
                        "requirement": decision.target_requirement,
                    },
                )

                state.questions.append(act_result.to_dict())
                state.covered_requirements.add(decision.target_requirement)
                
                yield {"type": "progress", "data": state.questions}

                current_covered = len(state.covered_requirements)
                if current_covered > prev_covered_count:
                    state.consecutive_no_progress = 0
                    prev_covered_count = current_covered
                else:
                    state.consecutive_no_progress += 1
            else:
                state.consecutive_no_progress += 1

        logger.info(
            "AgentLoop complete",
            extra={
                "questions_generated": len(state.questions),
                "requirements_covered": len(state.covered_requirements),
                "total_tokens": state.tokens_used,
                "iterations": state.iteration + 1,
            },
        )
        yield {"type": "done", "data": state.questions}
