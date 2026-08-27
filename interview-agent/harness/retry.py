"""Exponential backoff with jitter for LLM / tool calls."""

import logging
import random
import time
import re
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts have been exhausted."""


def with_retry(fn: Callable[[], T], retry_config: dict) -> T:
    """
    Execute *fn* with exponential backoff and uniform jitter.

    Parameters
    ----------
    fn:
        A zero-argument callable to attempt.
    retry_config:
        Dict with keys: max_attempts, base_delay_s, max_delay_s.

    Returns
    -------
    The return value of *fn* on the first successful call.

    Raises
    ------
    MaxRetriesExceeded
        If every attempt raises an exception.
    """
    max_attempts: int = int(retry_config.get("max_attempts", 3))
    base_delay: float = float(retry_config.get("base_delay_s", 1.0))
    max_delay: float = float(retry_config.get("max_delay_s", 30.0))

    last_exc: Exception = RuntimeError("No attempts made")
    
    # We allow more attempts if we are hitting API rate limits
    total_attempts_allowed = max_attempts + 5
    attempts_used = 0

    while attempts_used < total_attempts_allowed:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            attempts_used += 1
            
            err_str = str(exc)
            delay = None
            
            # Check for 429 quota/rate limit error and explicit retry delay
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                match = re.search(r"Please retry in (\d+\.?\d*)s", err_str)
                if match:
                    delay = float(match.group(1)) + 1.0  # Add 1s buffer
                else:
                    delay = max_delay  # Default to max delay if we can't parse it
                    
            if attempts_used < total_attempts_allowed:
                if delay is None:
                    # Normal exponential backoff for non-429 errors
                    delay = min(
                        base_delay * (2 ** (attempts_used - 1)) + random.uniform(0.0, 1.0),
                        max_delay,
                    )
                
                logger.warning(
                    "Retry attempt %d/%d failed — retrying in %.2fs",
                    attempts_used,
                    total_attempts_allowed,
                    delay,
                    extra={"error": err_str, "attempt": attempts_used},
                )
                time.sleep(delay)
            else:
                logger.error(
                    "All %d attempts failed.",
                    total_attempts_allowed,
                    extra={"error": err_str},
                )

    raise MaxRetriesExceeded(
        f"Failed after {max_attempts} attempt(s)"
    ) from last_exc
