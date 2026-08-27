"""
Tests for the reliability harness: guardrails, retry, fallback functions.
"""

import time
import pytest
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Guardrails tests
# ---------------------------------------------------------------------------

class TestGuardrails:

    def test_from_config_reads_values(self):
        from harness.guardrails import Guardrails
        config = {
            "agent": {
                "max_iterations": 15,
                "max_retries_per_question": 2,
                "stuck_loop_window": 4,
                "token_budget": 30_000,
            }
        }
        g = Guardrails.from_config(config)
        assert g.max_iterations == 15
        assert g.max_retries_per_question == 2
        assert g.stuck_loop_window == 4
        assert g.token_budget == 30_000

    def test_from_config_defaults(self):
        from harness.guardrails import Guardrails
        g = Guardrails.from_config({})
        assert g.max_iterations == 20
        assert g.token_budget == 50_000

    def test_check_token_budget_halt(self):
        from harness.guardrails import check_token_budget
        assert check_token_budget(50_001, 50_000) is True

    def test_check_token_budget_ok(self):
        from harness.guardrails import check_token_budget
        assert check_token_budget(1000, 50_000) is False

    def test_check_token_budget_at_limit(self):
        from harness.guardrails import check_token_budget
        assert check_token_budget(50_000, 50_000) is True

    def test_check_stuck_loop_halt(self):
        from harness.guardrails import check_stuck_loop
        assert check_stuck_loop(3, 3) is True

    def test_check_stuck_loop_ok(self):
        from harness.guardrails import check_stuck_loop
        assert check_stuck_loop(2, 3) is False

    def test_check_iteration_limit_halt(self):
        from harness.guardrails import check_iteration_limit
        assert check_iteration_limit(20, 20) is True

    def test_check_iteration_limit_ok(self):
        from harness.guardrails import check_iteration_limit
        assert check_iteration_limit(5, 20) is False

    def test_check_all_covered_true(self):
        from harness.guardrails import check_all_covered
        assert check_all_covered({"Python", "SQL"}, ["Python", "SQL"]) is True

    def test_check_all_covered_false(self):
        from harness.guardrails import check_all_covered
        assert check_all_covered({"Python"}, ["Python", "SQL"]) is False

    def test_check_all_covered_empty_requirements(self):
        from harness.guardrails import check_all_covered
        # Empty requirements list → not considered "all covered"
        assert check_all_covered(set(), []) is False


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------

class TestRetry:

    def test_succeeds_on_first_attempt(self):
        from harness.retry import with_retry
        fn = MagicMock(return_value="ok")
        result = with_retry(fn, {"max_attempts": 3, "base_delay_s": 0, "max_delay_s": 0})
        assert result == "ok"
        assert fn.call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        from harness.retry import with_retry
        fn = MagicMock(side_effect=[RuntimeError("fail"), RuntimeError("fail"), "ok"])
        result = with_retry(fn, {"max_attempts": 3, "base_delay_s": 0, "max_delay_s": 0})
        assert result == "ok"
        assert fn.call_count == 3

    def test_raises_after_max_attempts(self):
        from harness.retry import with_retry, MaxRetriesExceeded
        fn = MagicMock(side_effect=RuntimeError("always fails"))
        with pytest.raises(MaxRetriesExceeded):
            with_retry(fn, {"max_attempts": 2, "base_delay_s": 0, "max_delay_s": 0})
        assert fn.call_count == 2

    def test_does_not_sleep_with_zero_delay(self):
        """Verifies tests don't hang — zero delay config."""
        from harness.retry import with_retry
        fn = MagicMock(side_effect=[RuntimeError("x"), "ok"])
        t0 = time.time()
        with_retry(fn, {"max_attempts": 2, "base_delay_s": 0, "max_delay_s": 0})
        elapsed = time.time() - t0
        assert elapsed < 1.0  # should be near-instant


# ---------------------------------------------------------------------------
# Fallback function tests
# ---------------------------------------------------------------------------

class TestFallbacks:

    def test_handle_malformed_generation_returns_dict(self):
        from harness.fallback import handle_malformed_generation
        result = handle_malformed_generation("not json", "Python proficiency")
        assert "question" in result
        assert "answer_hint" in result
        assert "Python proficiency" in result["question"] or "Python proficiency" in result["target_skill"]

    def test_malformed_generation_uses_resume_evidence(self):
        from harness.fallback import handle_malformed_generation
        result = handle_malformed_generation(
            "not json",
            "Microservices architecture",
            {"evidence": [{"excerpt": "Led migration to 14 FastAPI microservices."}]},
        )
        assert "14 FastAPI microservices" in result["question"]
        assert result["resume_connection"] == "Led migration to 14 FastAPI microservices."

    def test_handle_malformed_reflection_returns_accept(self):
        from harness.fallback import handle_malformed_reflection
        result = handle_malformed_reflection("not json")
        assert result["verdict"] == "accept"
        assert "scores" in result

    def test_handle_extract_requirements_failure_returns_dict(self):
        from harness.fallback import handle_extract_requirements_failure
        result = handle_extract_requirements_failure("Some JD text")
        assert "required_skills" in result
        assert isinstance(result["required_skills"], list)

    def test_handle_find_resume_evidence_failure_no_evidence(self):
        from harness.fallback import handle_find_resume_evidence_failure
        result = handle_find_resume_evidence_failure("Python proficiency")
        assert result["has_evidence"] is False
        assert result["evidence"] == []

    def test_handle_memory_recall_failure_returns_empty_list(self):
        from harness.fallback import handle_memory_recall_failure
        result = handle_memory_recall_failure("some query")
        assert result == []

    def test_extract_json_from_text_fenced(self):
        from harness.fallback import extract_json_from_text
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result == {"key": "value"}

    def test_extract_json_from_text_raw(self):
        from harness.fallback import extract_json_from_text
        text = 'Here is the answer: {"key": "value"}'
        result = extract_json_from_text(text)
        assert result == {"key": "value"}

    def test_extract_json_from_text_invalid_returns_none(self):
        from harness.fallback import extract_json_from_text
        result = extract_json_from_text("this is not json at all")
        assert result is None

    def test_handle_malformed_reasoning_picks_first_uncovered(self):
        from harness.fallback import handle_malformed_reasoning
        uncovered = ["Distributed Systems", "SQL", "Kubernetes"]
        result = handle_malformed_reasoning("garbage", uncovered)
        assert result is not None
        assert result["target_requirement"] == "Distributed Systems"
        assert result["question_type"] == "jd_based"

    def test_handle_malformed_reasoning_empty_uncovered_returns_none(self):
        from harness.fallback import handle_malformed_reasoning
        result = handle_malformed_reasoning("garbage", [])
        assert result is None
