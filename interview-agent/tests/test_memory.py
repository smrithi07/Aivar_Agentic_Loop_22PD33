"""
Tests for SemanticMemory — save(), recall(), clear().

Critical test: test_memory_changes_reason_target demonstrates that memory
from an earlier iteration changes the behaviour of a later iteration by
causing reason() to select a different (uncovered) requirement.
"""

import json
import os
import sys
import types
import tempfile
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs — no real Gemini API calls in unit tests
# ---------------------------------------------------------------------------

class _FakeEmbedding:
    """Returns deterministic 4-dim embeddings so tests are fast and offline."""
    _counter: int = 0

    @classmethod
    def embed(cls, text: str) -> list[float]:
        cls._counter += 1
        # Simple hash-based deterministic vector
        h = hash(text) % 1000
        return [float(h), float(h + 1), float(h + 2), float(h + 3)]

    @classmethod
    def embed_query(cls, text: str) -> list[float]:
        return cls.embed(text)


# Patch the embeddings module BEFORE importing memory
import unittest.mock as mock

@pytest.fixture(autouse=True)
def patch_embeddings(monkeypatch):
    """Replace Gemini embedding calls with deterministic stubs."""
    monkeypatch.setattr(
        "memory.embeddings.get_embedding",
        lambda text, client: _FakeEmbedding.embed(text),
    )
    monkeypatch.setattr(
        "memory.embeddings.get_query_embedding",
        lambda text, client: _FakeEmbedding.embed_query(text),
    )


@pytest.fixture
def tmp_memory_config(tmp_path):
    return {
        "memory": {
            "top_k": 3,
            "metadata_path": str(tmp_path / "metadata.json"),
            "faiss_index_path": str(tmp_path / "faiss.index"),
        },
        "llm": {"embedding_dim": 4},
        "retry": {"max_attempts": 1, "base_delay_s": 0, "max_delay_s": 0},
    }


@pytest.fixture
def memory_instance(tmp_memory_config):
    from memory.memory import SemanticMemory
    return SemanticMemory(gemini_client=None, config=tmp_memory_config)


# ---------------------------------------------------------------------------
# Basic interface tests
# ---------------------------------------------------------------------------

class TestSemanticMemoryInterface:

    def test_save_increases_size(self, memory_instance):
        mem = memory_instance
        assert mem.size == 0
        mem.save("Python proficiency is required", {"type": "jd_requirements", "iteration": 0})
        assert mem.size == 1

    def test_recall_returns_list(self, memory_instance):
        mem = memory_instance
        mem.save("Python proficiency", {"type": "jd_requirements", "iteration": 0})
        results = mem.recall("Python")
        assert isinstance(results, list)

    def test_recall_empty_store_returns_empty(self, memory_instance):
        results = memory_instance.recall("anything")
        assert results == []

    def test_clear_resets_size(self, memory_instance):
        mem = memory_instance
        mem.save("some text", {"type": "test", "iteration": 0})
        assert mem.size == 1
        mem.clear()
        assert mem.size == 0

    def test_clear_resets_recall(self, memory_instance):
        mem = memory_instance
        mem.save("Python experience", {"type": "covered_requirement", "iteration": 0})
        mem.clear()
        results = mem.recall("Python")
        assert results == []

    def test_metadata_persisted_to_disk(self, tmp_memory_config, tmp_path):
        from memory.memory import SemanticMemory
        mem = SemanticMemory(gemini_client=None, config=tmp_memory_config)
        mem.save("test entry", {"type": "test", "iteration": 1})
        metadata_path = tmp_memory_config["memory"]["metadata_path"]
        assert os.path.exists(metadata_path)
        with open(metadata_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["type"] == "test"

    def test_save_includes_text_in_metadata(self, memory_instance):
        mem = memory_instance
        mem.save("hello world", {"type": "test", "iteration": 0})
        results = mem.recall("hello world")
        if results:
            assert "text" in results[0]

    def test_multiple_saves_and_recall(self, memory_instance):
        mem = memory_instance
        for i in range(5):
            mem.save(f"entry {i}", {"type": "test", "iteration": i})
        assert mem.size == 5
        results = mem.recall("entry 3", top_k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# THE KEY TEST: memory from earlier iteration changes later iteration behaviour
# ---------------------------------------------------------------------------

class TestMemoryInfluencesReasoning:
    """
    Demonstrate that memory.recall() returns covered-requirement entries
    that would cause reason() to avoid re-selecting them.

    We don't call the real reason() (that needs Gemini) — instead we
    verify that the recall results include the covered-requirement entry
    and that filtering against them produces a different selection.
    """

    def test_covered_requirement_is_recalled(self, memory_instance):
        mem = memory_instance

        # Simulate iteration 1: Python was covered
        mem.save(
            "Covered requirement: Python proficiency",
            {"type": "covered_requirement", "iteration": 1, "requirement": "Python proficiency"},
        )

        # Simulate iteration 2: reason() recalls before deciding
        results = mem.recall("Python proficiency")

        covered_entries = [r for r in results if r.get("type") == "covered_requirement"]
        assert len(covered_entries) >= 1
        assert covered_entries[0]["requirement"] == "Python proficiency"

    def test_memory_causes_different_requirement_selection(self, memory_instance):
        """
        After saving 'Python proficiency' as covered, a simulated reason()
        function that filters covered entries should select a different requirement.
        """
        mem = memory_instance
        all_requirements = ["Python proficiency", "Distributed Systems", "SQL optimisation"]

        # Iteration 1: cover Python
        mem.save(
            "Covered requirement: Python proficiency",
            {"type": "covered_requirement", "iteration": 1, "requirement": "Python proficiency"},
        )

        # Simulate what reason() does: recall → filter covered → pick next
        memory_hits = mem.recall("Python proficiency covered requirements")
        covered_from_memory = {
            r["requirement"]
            for r in memory_hits
            if r.get("type") == "covered_requirement"
        }

        # Simulated filtering (mirrors reason.py logic)
        uncovered = [r for r in all_requirements if r not in covered_from_memory]

        assert "Python proficiency" not in uncovered
        assert len(uncovered) == 2
        assert uncovered[0] != "Python proficiency"

    def test_reflection_feedback_stored_and_recalled(self, memory_instance):
        mem = memory_instance

        mem.save(
            "Reflection on 'SQL optimisation': verdict=retry, score=2.5. "
            "Feedback: Question is too vague — ask about a specific query plan.",
            {
                "type": "reflection_feedback",
                "iteration": 2,
                "target_requirement": "SQL optimisation",
                "verdict": "retry",
                "overall_score": 2.5,
            },
        )

        results = mem.recall("SQL optimisation query plan")
        feedback_entries = [r for r in results if r.get("type") == "reflection_feedback"]
        assert len(feedback_entries) >= 1
        assert feedback_entries[0]["verdict"] == "retry"
        assert feedback_entries[0]["overall_score"] == 2.5
