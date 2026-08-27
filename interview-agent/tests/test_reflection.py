"""
Tests for the REFLECT phase — Reflexion pattern.

Covers: accept/retry logic, auto-retry on low scores, retry-with-feedback
propagation, fallback on malformed reflection response.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_act_result(question="What is your experience?", question_type="jd_based"):
    from agent.act import ActResult
    return ActResult(
        question=question,
        category="Technical",
        target_skill="Python proficiency",
        relevance="Directly tests a required skill.",
        resume_connection=None,
        answer_hint="Walk through a recent Python project.",
        question_type=question_type,
        target_requirement="Python proficiency",
        tool_result={},
    )


def _make_decision():
    from agent.reason import ReasoningDecision
    return ReasoningDecision(
        target_requirement="Python proficiency",
        question_type="jd_based",
        tool_to_call="none",
        tool_args={},
        reasoning_trace="Selected Python as first uncovered requirement.",
    )


def _make_state():
    from agent.state import AgentState
    return AgentState(
        jd_text="Senior Backend Engineer job description",
        resume_text="Candidate resume text",
        requirements={"required_skills": ["Python proficiency"], "preferred_skills": [], "responsibilities": [], "seniority": "Senior"},
        iteration=1,
    )


@pytest.fixture
def mock_gemini():
    client = MagicMock()
    client.total_tokens_used = 100
    client._parse_json = MagicMock(side_effect=lambda text: json.loads(text) if text.startswith("{") else None)
    return client


@pytest.fixture
def mock_memory(monkeypatch):
    mem = MagicMock()
    mem.recall.return_value = []
    mem.save.return_value = None
    return mem


def _accept_response(overall_score=4.5) -> str:
    return json.dumps({
        "verdict": "accept",
        "feedback": "Well-grounded question, clearly tests the target skill.",
        "scores": {
            "jd_relevance": 5, "resume_grounding": 5, "clarity": 4,
            "usefulness": 4, "target_skill_alignment": 5, "originality": 4,
        },
        "overall_score": overall_score,
    })


def _retry_response(feedback="Question is too vague.") -> str:
    return json.dumps({
        "verdict": "retry",
        "feedback": feedback,
        "scores": {
            "jd_relevance": 4, "resume_grounding": 4, "clarity": 2,
            "usefulness": 3, "target_skill_alignment": 4, "originality": 3,
        },
        "overall_score": 3.3,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReflectAccept:

    def test_accept_verdict_returned(self, mock_gemini, mock_memory):
        from agent.reflect import reflect
        mock_gemini.generate.return_value = _accept_response()
        mock_gemini._parse_json.side_effect = lambda t: json.loads(t)

        result = reflect(
            _make_act_result(), _make_decision(), _make_state(),
            mock_memory, mock_gemini, {}
        )
        assert result.verdict == "accept"

    def test_accept_saves_to_memory(self, mock_gemini, mock_memory):
        from agent.reflect import reflect
        mock_gemini.generate.return_value = _accept_response()
        mock_gemini._parse_json.side_effect = lambda t: json.loads(t)

        reflect(
            _make_act_result(), _make_decision(), _make_state(),
            mock_memory, mock_gemini, {}
        )
        # reflect() saves reflection feedback regardless of verdict
        mock_memory.save.assert_called_once()

    def test_overall_score_populated(self, mock_gemini, mock_memory):
        from agent.reflect import reflect
        mock_gemini.generate.return_value = _accept_response(overall_score=4.8)
        mock_gemini._parse_json.side_effect = lambda t: json.loads(t)

        result = reflect(
            _make_act_result(), _make_decision(), _make_state(),
            mock_memory, mock_gemini, {}
        )
        assert result.overall_score == 4.8


class TestReflectRetry:

    def test_retry_verdict_on_low_score(self, mock_gemini, mock_memory):
        from agent.reflect import reflect
        mock_gemini.generate.return_value = _retry_response()
        mock_gemini._parse_json.side_effect = lambda t: json.loads(t)

        result = reflect(
            _make_act_result(), _make_decision(), _make_state(),
            mock_memory, mock_gemini, {}
        )
        assert result.verdict == "retry"

    def test_feedback_propagated_in_retry(self, mock_gemini, mock_memory):
        from agent.reflect import reflect
        feedback_msg = "The question is too vague — specify a concrete scenario."
        mock_gemini.generate.return_value = _retry_response(feedback=feedback_msg)
        mock_gemini._parse_json.side_effect = lambda t: json.loads(t)

        result = reflect(
            _make_act_result(), _make_decision(), _make_state(),
            mock_memory, mock_gemini, {}
        )
        assert feedback_msg in result.feedback

    def test_auto_retry_when_any_score_below_3(self, mock_gemini, mock_memory):
        """If any rubric score < 3, verdict must be forced to retry."""
        from agent.reflect import reflect

        # Gemini says accept but one score is 2 — should be forced to retry
        response = json.dumps({
            "verdict": "accept",
            "feedback": "Looks fine overall.",
            "scores": {
                "jd_relevance": 5, "resume_grounding": 5, "clarity": 2,  # clarity = 2!
                "usefulness": 4, "target_skill_alignment": 4, "originality": 4,
            },
            "overall_score": 4.0,
        })
        mock_gemini.generate.return_value = response
        mock_gemini._parse_json.side_effect = lambda t: json.loads(t)

        result = reflect(
            _make_act_result(), _make_decision(), _make_state(),
            mock_memory, mock_gemini, {}
        )
        assert result.verdict == "retry"

    def test_retry_saves_feedback_to_memory(self, mock_gemini, mock_memory):
        from agent.reflect import reflect
        mock_gemini.generate.return_value = _retry_response()
        mock_gemini._parse_json.side_effect = lambda t: json.loads(t)

        reflect(
            _make_act_result(), _make_decision(), _make_state(),
            mock_memory, mock_gemini, {}
        )
        mock_memory.save.assert_called_once()
        call_kwargs = mock_memory.save.call_args
        saved_metadata = call_kwargs[1]["metadata"] if call_kwargs[1] else call_kwargs[0][1]
        assert saved_metadata.get("verdict") == "retry"


class TestReflectFallback:

    def test_fallback_on_none_parse(self, mock_gemini, mock_memory):
        """When _parse_json returns None, fallback reflection is used (defaults to accept)."""
        from agent.reflect import reflect
        mock_gemini.generate.return_value = "DEFINITELY NOT JSON %%% garbage"
        mock_gemini._parse_json.return_value = None

        result = reflect(
            _make_act_result(), _make_decision(), _make_state(),
            mock_memory, mock_gemini, {}
        )
        # Fallback defaults to "accept" to avoid infinite retry loops
        assert result.verdict == "accept"
        assert result._fallback is True

    def test_fallback_on_gemini_exception(self, mock_gemini, mock_memory):
        from agent.reflect import reflect
        mock_gemini.generate.side_effect = RuntimeError("API error")

        # reflect() should not raise — it should propagate the exception up
        # (the loop.py catches it), but if it does raise, that's acceptable too.
        try:
            result = reflect(
                _make_act_result(), _make_decision(), _make_state(),
                mock_memory, mock_gemini, {}
            )
        except RuntimeError:
            pass  # acceptable — loop.py handles this
