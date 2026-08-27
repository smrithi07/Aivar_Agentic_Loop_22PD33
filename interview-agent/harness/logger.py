"""Structured JSON logger for the InterviewLens agent."""

import json
import logging
import sys
import time
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any extra fields passed via the `extra` kwarg
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "message", "module", "msecs", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                log_obj[key] = value
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


def setup_logger(config: dict) -> logging.Logger:
    """Configure the root logger from config.yaml settings."""
    level_name: str = config.get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers (important during tests)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    log_file: Optional[str] = config.get("logging", {}).get("log_file")
    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(level)
        fh.setFormatter(JSONFormatter())
        root.addHandler(fh)

    return logging.getLogger("interview_agent")


def log_step(
    logger: logging.Logger,
    step: str,
    iteration: int,
    inputs: dict,
    outputs: dict,
    duration_ms: float = 0.0,
    tokens_used: int = 0,
    error: Optional[str] = None,
) -> None:
    """Emit a structured log entry for one agent step."""
    logger.info(
        "agent_step",
        extra={
            "step": step,
            "iteration": iteration,
            "duration_ms": round(duration_ms, 2),
            "inputs": inputs,
            "outputs": outputs,
            "tokens_used": tokens_used,
            "error": error,
        },
    )


class StepTimer:
    """Context manager that measures elapsed milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "StepTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
