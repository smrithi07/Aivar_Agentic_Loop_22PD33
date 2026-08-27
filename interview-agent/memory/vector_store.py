"""
FAISSStore — in-process exact L2 vector store.

Uses faiss.IndexFlatL2 (no approximation, correct for single-session scale).
Indices are kept in memory; reset() clears everything.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed — vector search unavailable")


class FAISSStore:
    """
    Exact L2 nearest-neighbour vector store backed by faiss.IndexFlatL2.

    Each stored vector has an integer ID that maps 1-to-1 to a metadata
    entry in SemanticMemory.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index: Optional[object] = None
        self._count: int = 0
        self._build_index()

    def _build_index(self) -> None:
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatL2(self.dim)
        self._count = 0

    def add(self, vector: list[float]) -> int:
        """
        Add *vector* to the store.
        Returns the 0-based integer ID assigned to this entry.
        """
        if not FAISS_AVAILABLE or self._index is None:
            idx = self._count
            self._count += 1
            return idx

        arr = np.array([vector], dtype=np.float32)
        self._index.add(arr)  # type: ignore[union-attr]
        idx = self._count
        self._count += 1
        return idx

    def search(self, query_vector: list[float], top_k: int) -> list[int]:
        """
        Return the IDs of the *top_k* most similar stored vectors.
        Returns an empty list if the store is empty or FAISS is unavailable.
        """
        if not FAISS_AVAILABLE or self._index is None or self._count == 0:
            return []

        k = min(top_k, self._count)
        arr = np.array([query_vector], dtype=np.float32)
        _distances, indices = self._index.search(arr, k)  # type: ignore[union-attr]
        return [int(i) for i in indices[0] if i >= 0]

    def reset(self) -> None:
        """Discard all stored vectors and reset the index."""
        self._build_index()

    @property
    def size(self) -> int:
        return self._count
