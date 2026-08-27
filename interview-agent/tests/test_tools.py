"""
Tests for extract_requirements and find_resume_evidence tools.

Uses mock GeminiClient to avoid real API calls.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gemini():
    client = MagicMock()
    client.total_tokens_used = 0
    return client


@pytest.fixture
def sample_jd():
    return """
Senior Backend Engineer
Required: Python, REST API design, SQL, distributed systems experience
Preferred: Kubernetes, Kafka
Responsibilities: Design microservices, mentor junior engineers
Seniority: Senior
    """.strip()


@pytest.fixture
def sample_resume():
    return """
Jane Doe | Backend Engineer

Experience:
- Led migration of monolith to 12 microservices using Python FastAPI at TechCorp (2021-2024).
- Deployed services on GKE across 3 regions; authored Helm charts.
- Designed PostgreSQL schemas for multi-tenant SaaS product; optimised queries from 8s to 200ms.
- Built Kafka pipeline processing 5M events/day.

Skills: Python, FastAPI, PostgreSQL, Redis, Kubernetes, Kafka, Docker, GCP
    """.strip()


# ---------------------------------------------------------------------------
# extract_requirements tests
# ---------------------------------------------------------------------------

class TestExtractRequirements:

    def test_returns_dict_with_required_keys(self, mock_gemini, sample_jd):
        from tools.extract_requirements import extract_requirements

        mock_gemini.generate.return_value = json.dumps({
            "required_skills": ["Python", "REST API design", "SQL"],
            "preferred_skills": ["Kubernetes"],
            "responsibilities": ["Design microservices"],
            "seniority": "Senior",
        })

        result = extract_requirements(sample_jd, mock_gemini)

        assert "required_skills" in result
        assert "preferred_skills" in result
        assert "responsibilities" in result
        assert "seniority" in result

    def test_required_skills_is_list(self, mock_gemini, sample_jd):
        from tools.extract_requirements import extract_requirements

        mock_gemini.generate.return_value = json.dumps({
            "required_skills": ["Python", "SQL"],
            "preferred_skills": [],
            "responsibilities": ["Design systems"],
            "seniority": "Senior",
        })

        result = extract_requirements(sample_jd, mock_gemini)
        assert isinstance(result["required_skills"], list)

    def test_fallback_on_empty_jd(self, mock_gemini):
        from tools.extract_requirements import extract_requirements

        result = extract_requirements("", mock_gemini)
        assert "required_skills" in result

    def test_fallback_on_malformed_response(self, mock_gemini, sample_jd):
        from tools.extract_requirements import extract_requirements

        mock_gemini.generate.return_value = "I cannot parse this properly!!! ###"

        result = extract_requirements(sample_jd, mock_gemini)
        # Should return a fallback dict, not raise
        assert isinstance(result, dict)
        assert "required_skills" in result

    def test_fallback_on_gemini_exception(self, mock_gemini, sample_jd):
        from tools.extract_requirements import extract_requirements

        mock_gemini.generate.side_effect = RuntimeError("API timeout")

        result = extract_requirements(sample_jd, mock_gemini)
        assert isinstance(result, dict)

    def test_seniority_preserved(self, mock_gemini, sample_jd):
        from tools.extract_requirements import extract_requirements

        mock_gemini.generate.return_value = json.dumps({
            "required_skills": ["Python"],
            "preferred_skills": [],
            "responsibilities": [],
            "seniority": "Staff",
        })

        result = extract_requirements(sample_jd, mock_gemini)
        assert result["seniority"] == "Staff"

    def test_json_in_markdown_fence_is_parsed(self, mock_gemini, sample_jd):
        from tools.extract_requirements import extract_requirements

        # Simulate Gemini wrapping JSON in a markdown fence
        mock_gemini.generate.return_value = (
            "Sure! Here is the result:\n```json\n"
            + json.dumps({
                "required_skills": ["Python"],
                "preferred_skills": ["Docker"],
                "responsibilities": ["Build APIs"],
                "seniority": "Mid",
            })
            + "\n```"
        )

        result = extract_requirements(sample_jd, mock_gemini)
        assert result["required_skills"] == ["Python"]


# ---------------------------------------------------------------------------
# find_resume_evidence tests
# ---------------------------------------------------------------------------

class TestFindResumeEvidence:

    def setup_method(self):
        """Clear the module-level resume cache before each test."""
        from tools.find_resume_evidence import clear_resume_cache
        clear_resume_cache()

    def _make_embed_client(self, dim: int = 4):
        client = MagicMock()
        client.total_tokens_used = 0
        import random
        client.embed.side_effect = lambda text: [random.random() for _ in range(dim)]
        client.embed_query.side_effect = lambda text: [random.random() for _ in range(dim)]
        return client

    def test_returns_dict_with_expected_keys(self, sample_resume):
        from tools.find_resume_evidence import find_resume_evidence

        client = self._make_embed_client()
        result = find_resume_evidence("Python", sample_resume, client, top_k=2)

        assert "evidence" in result
        assert "has_evidence" in result

    def test_has_evidence_is_bool(self, sample_resume):
        from tools.find_resume_evidence import find_resume_evidence

        client = self._make_embed_client()
        result = find_resume_evidence("Kubernetes", sample_resume, client, top_k=2)
        assert isinstance(result["has_evidence"], bool)

    def test_evidence_is_list(self, sample_resume):
        from tools.find_resume_evidence import find_resume_evidence

        client = self._make_embed_client()
        result = find_resume_evidence("FastAPI", sample_resume, client, top_k=2)
        assert isinstance(result["evidence"], list)

    def test_fallback_on_empty_resume(self):
        from tools.find_resume_evidence import find_resume_evidence, clear_resume_cache
        clear_resume_cache()

        client = self._make_embed_client()
        result = find_resume_evidence("Python", "", client)
        assert result["has_evidence"] is False

    def test_fallback_on_empty_requirement(self, sample_resume):
        from tools.find_resume_evidence import find_resume_evidence, clear_resume_cache
        clear_resume_cache()

        client = self._make_embed_client()
        result = find_resume_evidence("", sample_resume, client)
        assert result["has_evidence"] is False

    def test_fallback_on_embedding_error(self, sample_resume):
        from tools.find_resume_evidence import find_resume_evidence, clear_resume_cache
        clear_resume_cache()

        client = MagicMock()
        client.embed.side_effect = RuntimeError("embedding failed")
        client.embed_query.side_effect = RuntimeError("embedding failed")

        result = find_resume_evidence("Python", sample_resume, client)
        assert "has_evidence" in result
        assert result["has_evidence"] is False

    def test_resume_index_is_cached(self, sample_resume):
        """The resume should only be embedded once across multiple calls."""
        from tools.find_resume_evidence import find_resume_evidence, clear_resume_cache
        clear_resume_cache()

        client = self._make_embed_client()
        embed_calls_before = client.embed.call_count

        find_resume_evidence("Python", sample_resume, client)
        calls_after_first = client.embed.call_count

        find_resume_evidence("Kubernetes", sample_resume, client)
        calls_after_second = client.embed.call_count

        # Second call should NOT re-embed the resume chunks
        # (only the query is embedded on subsequent calls)
        assert calls_after_second < calls_after_first * 2
