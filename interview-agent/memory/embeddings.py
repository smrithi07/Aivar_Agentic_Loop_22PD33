"""
Embedding helpers — thin wrapper so the rest of the codebase
never imports google-generativeai directly for embeddings.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_embedding(text: str, gemini_client) -> list[float]:
    """
    Return a document embedding for *text* using the configured Gemini
    embedding model.  Delegates to GeminiClient.embed().
    """
    return gemini_client.embed(text)


def get_query_embedding(text: str, gemini_client) -> list[float]:
    """
    Return a query embedding for *text*.
    Uses asymmetric retrieval_query task type for better recall.
    """
    return gemini_client.embed_query(text)
