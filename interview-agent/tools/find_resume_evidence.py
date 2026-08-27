"""
Tool 2 — find_resume_evidence

Given a JD requirement string, returns the most relevant excerpts from the
candidate's resume using Gemini embeddings + FAISS similarity search.

Caches the resume FAISS index across calls within the same run so the resume
is only chunked and embedded once.
"""

import logging
import re
from typing import Optional

import numpy as np

from harness.fallback import handle_find_resume_evidence_failure
from memory.vector_store import FAISSStore

logger = logging.getLogger(__name__)

# Module-level cache — reset by calling clear_resume_cache()
_resume_cache: dict = {
    "chunks": [],      # list[str] — resume text chunks
    "store": None,     # FAISSStore | None
    "resume_text": "", # the resume text that was indexed
}


def find_resume_evidence(
    requirement: str,
    resume_text: str,
    gemini_client,
    top_k: int = 3,
) -> dict:
    """
    Find the *top_k* most relevant resume excerpts for *requirement*.

    Parameters
    ----------
    requirement : str
        The specific JD requirement to search for.
    resume_text : str
        Full candidate resume text.
    gemini_client : GeminiClient
        Configured Gemini client (handles retry internally).
    top_k : int
        Number of evidence snippets to return.

    Returns
    -------
    dict matching FIND_RESUME_EVIDENCE_RETURN schema.
    Falls back to handle_find_resume_evidence_failure() on any error.
    """
    if not requirement or not resume_text:
        return handle_find_resume_evidence_failure(requirement or "")

    try:
        _ensure_index(resume_text, gemini_client)

        store: FAISSStore = _resume_cache["store"]
        chunks: list[str] = _resume_cache["chunks"]

        if store is None or store.size == 0 or not chunks:
            logger.warning("Resume index is empty — no evidence available")
            return {"evidence": [], "has_evidence": False}

        # Embed the requirement as a query
        query_vec = gemini_client.embed_query(requirement)
        ids = store.search(query_vec, top_k=top_k)

        evidence = []
        for idx in ids:
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                evidence.append({
                    "section": _infer_section(chunk),
                    "excerpt": chunk,
                    "relevance_score": round(1.0 - (idx * 0.05), 2),  # rank-based proxy
                })

        result = {
            "evidence": evidence,
            "has_evidence": len(evidence) > 0,
        }

        logger.info(
            "find_resume_evidence succeeded",
            extra={
                "requirement": requirement[:60],
                "n_evidence": len(evidence),
                "has_evidence": result["has_evidence"],
            },
        )
        return result

    except Exception as exc:  # noqa: BLE001
        logger.error("find_resume_evidence failed: %s", exc)
        return handle_find_resume_evidence_failure(requirement)


def clear_resume_cache() -> None:
    """Reset the module-level resume index (called at start of each run)."""
    global _resume_cache
    _resume_cache = {"chunks": [], "store": None, "resume_text": ""}
    logger.debug("Resume evidence cache cleared")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_index(resume_text: str, gemini_client) -> None:
    """Build and cache the FAISS index for *resume_text* if not yet built."""
    if _resume_cache["resume_text"] == resume_text and _resume_cache["store"] is not None:
        return  # already indexed this resume

    logger.info("Building resume FAISS index ...")
    chunks = _chunk_resume(resume_text)
    if not chunks:
        _resume_cache.update({"chunks": [], "store": FAISSStore(dim=768), "resume_text": resume_text})
        return

    # Embed all chunks (one by one to respect rate limits; retry is inside embed())
    vectors = []
    for chunk in chunks:
        vec = gemini_client.embed(chunk)
        vectors.append(vec)

    dim = len(vectors[0])
    store = FAISSStore(dim=dim)
    for vec in vectors:
        store.add(vec)

    _resume_cache.update({"chunks": chunks, "store": store, "resume_text": resume_text})
    logger.info("Resume index built with %d chunks", len(chunks))


def _chunk_resume(text: str, min_words: int = 8) -> list[str]:
    """
    Split the resume into meaningful chunks.

    Strategy:
    - Split on double newlines first (paragraph-level)
    - Further split long paragraphs on single newlines (bullet points)
    - Filter out very short fragments (< min_words words)
    """
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks = []
    for para in paragraphs:
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        for line in lines:
            word_count = len(line.split())
            if word_count >= min_words:
                chunks.append(line)
    return chunks


_SECTION_KEYWORDS = {
    "experience": ["experience", "work", "employment", "career", "role", "position"],
    "education": ["education", "university", "degree", "bachelor", "master", "phd", "college"],
    "skills": ["skill", "technology", "tool", "language", "framework", "stack"],
    "projects": ["project", "built", "developed", "created", "launched", "shipped"],
    "certifications": ["certification", "certified", "certificate", "aws", "gcp", "azure"],
}


def _infer_section(chunk: str) -> str:
    """Heuristically infer which resume section a chunk belongs to."""
    lower = chunk.lower()
    for section, keywords in _SECTION_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return section.capitalize()
    return "General"
