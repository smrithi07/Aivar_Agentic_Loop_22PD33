"""
SemanticMemory — persistent semantic memory for InterviewLens.

Architecture
------------
- FAISS IndexFlatL2 for vector similarity search (in-process)
- Flat JSON array in memory/metadata.json for rich metadata per entry
- Gemini embeddings for both save() and recall()

Memory is NOT conversational memory.  Within a single run it allows
later reasoning iterations to retrieve relevant prior context:

  reason()  → recall(query)    — reads before making decisions
  reflect() → save(text, meta) — writes after accepting a question

This prevents duplicate questions and allows requirement-coverage
awareness across all iterations of the agentic loop.

Entry types stored
------------------
  jd_requirements      — structured requirements from extract_requirements
  resume_evidence      — evidence snippet from find_resume_evidence
  generated_question   — accepted question + target skill
  covered_requirement  — signals that a requirement has been addressed
  reflection_feedback  — Reflexion verdict + feedback for future reference
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from harness.fallback import handle_memory_recall_failure, handle_memory_save_failure
from memory.embeddings import get_embedding, get_query_embedding
from memory.vector_store import FAISSStore

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Semantic memory store with save(), recall(), and clear() interface.

    Parameters
    ----------
    gemini_client : GeminiClient
        Used to compute embeddings.
    config : dict
        Full config dict; reads memory.* and llm.embedding_dim keys.
    """

    def __init__(self, gemini_client, config: dict) -> None:
        self._client = gemini_client
        mem_cfg = config.get("memory", {})
        llm_cfg = config.get("llm", {})

        self._dim: int = int(llm_cfg.get("embedding_dim", 3072))
        self._top_k: int = int(mem_cfg.get("top_k", 3))
        self._metadata_path: str = mem_cfg.get(
            "metadata_path", "memory/metadata.json"
        )

        # _store is lazy-initialised on the first embed call so the actual
        # embedding dimension (from the model) determines index size — avoiding
        # any config/model mismatch.
        self._store: Optional[FAISSStore] = None
        self._metadata: list[dict] = []

        # Load existing metadata from disk (supports cross-run persistence)
        self._load_metadata()

    def _get_store(self, dim: int) -> FAISSStore:
        """Return the FAISS store, creating it with *dim* on first call."""
        if self._store is None:
            self._store = FAISSStore(dim=dim)
        return self._store

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def save(self, text: str, metadata: dict) -> None:
        """
        Embed *text* and store it together with *metadata*.

        Metadata should include at minimum:
            type (str)  — entry type, e.g. "covered_requirement"
            iteration (int) — current loop iteration
        """
        try:
            vector = get_embedding(text, self._client)
            store = self._get_store(dim=len(vector))
            faiss_idx = store.add(vector)
            entry = {
                **metadata,
                "text": text,
                "faiss_index": faiss_idx,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            self._metadata.append(entry)
            self._persist_metadata()
            logger.debug(
                "Memory saved",
                extra={"type": metadata.get("type"), "faiss_idx": faiss_idx},
            )
        except Exception as exc:  # noqa: BLE001
            handle_memory_save_failure(text, metadata)

    def recall(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Find the *top_k* most semantically similar entries to *query*.

        Returns a list of metadata dicts (empty list if store is empty or
        embedding fails).
        """
        k = top_k if top_k is not None else self._top_k
        try:
            if self._store is None or self._store.size == 0:
                return []
            query_vec = get_query_embedding(query, self._client)
            ids = self._store.search(query_vec, k)
            results = []
            for idx in ids:
                if 0 <= idx < len(self._metadata):
                    results.append(self._metadata[idx])
            logger.debug(
                "Memory recalled",
                extra={"query": query[:80], "n_results": len(results)},
            )
            return results
        except Exception as exc:  # noqa: BLE001
            return handle_memory_recall_failure(query)

    def clear(self) -> None:
        """
        Reset both the FAISS index and the JSON metadata store.
        Called at the start of every run for a clean slate.
        """
        if self._store is not None:
            self._store.reset()
        self._store = None  # will be re-created with correct dim on next save
        self._metadata = []
        self._persist_metadata()
        logger.info("SemanticMemory cleared")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_metadata(self) -> None:
        """Write the metadata list to disk as a JSON array."""
        try:
            os.makedirs(os.path.dirname(self._metadata_path), exist_ok=True)
            with open(self._metadata_path, "w", encoding="utf-8") as fh:
                json.dump(self._metadata, fh, indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist metadata: %s", exc)

    def _load_metadata(self) -> None:
        """Load metadata from disk if the file exists (supports resumability)."""
        if not os.path.exists(self._metadata_path):
            return
        try:
            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list) and data:
                self._metadata = data
                logger.debug("Loaded %d metadata entries from disk", len(data))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load existing metadata: %s", exc)

    @property
    def size(self) -> int:
        return len(self._metadata)
