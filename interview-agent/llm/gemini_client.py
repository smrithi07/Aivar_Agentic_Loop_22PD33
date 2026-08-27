"""
GeminiClient — wrapper around the new google-genai SDK.

Uses `google.genai` (not the deprecated `google.generativeai`).

Responsibilities:
- Configure the Gemini SDK with the API key from .env
- Expose generate() for text/JSON generation
- Expose embed() / embed_query() for semantic embeddings
- Track cumulative token usage across all calls
- Wrap every API call with with_retry() from the harness
"""

import json
import logging
import os
import re
from typing import Optional

from google import genai
from google.genai import types

from harness.retry import with_retry

logger = logging.getLogger(__name__)


class GeminiClient:
    """Single-instance Gemini API client shared across all agent components."""

    def __init__(self, config: dict) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and fill in your key."
            )

        self._client = genai.Client(api_key=api_key)

        llm_cfg = config.get("llm", {})
        self.model_name: str = llm_cfg.get("model", "gemini-3.5-flash-lite")
        self.embedding_model: str = llm_cfg.get(
            "embedding_model", "gemini-embedding-2-preview"
        )
        self.temperature: float = float(llm_cfg.get("temperature", 0.7))
        self.max_output_tokens: int = int(llm_cfg.get("max_output_tokens", 4096))
        self.retry_config: dict = config.get("retry", {})

        self._generate_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        self.total_tokens_used: int = 0

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Send *prompt* to Gemini and return the raw text response.
        Retries on transient errors using exponential backoff.
        """

        def _call() -> str:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._generate_config,
            )
            self._track_tokens(response)
            if response.text is None:
                raise ValueError("Gemini returned no text content (response.text is None)")
            return response.text

        return with_retry(_call, self.retry_config)

    def generate_json(self, prompt: str) -> Optional[dict | list]:
        """
        Generate content expected to be a JSON object.
        Attempts to extract JSON from prose/markdown wrapping.
        Returns parsed dict or None on failure.
        """
        raw = self.generate(prompt)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed *text* as a retrieval document."""

        def _call() -> list[float]:
            return self._embed_text(text, task_type="RETRIEVAL_DOCUMENT")

        return with_retry(_call, self.retry_config)

    def embed_query(self, text: str) -> list[float]:
        """Embed *text* as a retrieval query (asymmetric embedding)."""

        def _call() -> list[float]:
            return self._embed_text(text, task_type="RETRIEVAL_QUERY")

        return with_retry(_call, self.retry_config)

    def _embed_text(self, text: str, task_type: str) -> list[float]:
        """
        Call the embedding model. Tries with task_type first; if the model
        rejects it (e.g. gemini-embedding-2-preview), retries without it.
        """
        try:
            result = self._client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            if not result.embeddings:
                raise ValueError("Embedding API returned no embeddings (result.embeddings is None or empty)")
            values = result.embeddings[0].values
            if values is None:
                raise ValueError("Embedding API returned no values")
            return values
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ("task_type", "unsupported", "invalid argument", "unknown")):
                # Retry without task_type for models that don't support it
                result = self._client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                )
                if not result.embeddings:
                    raise ValueError("Embedding API returned no embeddings (result.embeddings is None or empty)")
                values = result.embeddings[0].values
                if values is None:
                    raise ValueError("Embedding API returned no values")
                return values
            raise

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def _track_tokens(self, response: object) -> None:
        try:
            meta = getattr(response, "usage_metadata", None)
            if meta:
                count = (
                    getattr(meta, "total_token_count", None)
                    or getattr(meta, "candidates_token_count", 0)
                    or 0
                )
                self.total_tokens_used += count
        except Exception:  # noqa: BLE001
            pass  # non-fatal

    # ------------------------------------------------------------------
    # JSON extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Optional[dict | list]:
        """
        Extract and parse JSON from an LLM response that may be wrapped
        in markdown fences or preceded by prose.
        """
        if not text:
            return None

        text = text.strip()

        # 1. Try parsing directly (often works if LLM is well-behaved)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Try ```json ... ``` block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            candidate = match.group(1).strip()
        else:
            # 3. Try to find the outermost [...] or {...}
            match = re.search(r"([\[\{][\s\S]*[\]\}])", text)
            candidate = match.group(1).strip() if match else text

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            logger.warning(
                "JSON parse failed",
                extra={"snippet": text[:300]},
            )
            return None

