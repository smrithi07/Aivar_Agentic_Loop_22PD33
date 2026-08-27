"""
InterviewLens — main entry point.

Usage:
    python main.py [--jd path/to/jd.txt] [--resume path/to/resume.txt]
                   [--config config.yaml] [--output output/questions.json]
                   [--max-iterations N]

Defaults to data/sample_jd.txt and data/sample_resume.txt when no paths given.
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from datetime import datetime, timezone

# Load .env before any other imports that use env vars
from dotenv import load_dotenv
load_dotenv()

import yaml

from harness.logger import setup_logger
from llm.gemini_client import GeminiClient
from memory.memory import SemanticMemory
from agent.loop import AgentLoop


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InterviewLens — Agentic interview question generator"
    )
    parser.add_argument(
        "--jd",
        default="data/sample_jd.txt",
        help="Path to the job description text file",
    )
    parser.add_argument(
        "--resume",
        default="data/sample_resume.txt",
        help="Path to the candidate resume text file",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save questions as JSON (e.g. output/questions.json)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override agent.max_iterations from config",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_text(path: str, label: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    content = p.read_text(encoding="utf-8").strip()
    if not content:
        print(f"[ERROR] {label} file is empty: {path}", file=sys.stderr)
        sys.exit(1)
    return content


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_wrapped(label: str, value: object, width: int = 78) -> None:
    """Print a labelled value with consistent wrapping for terminal demos."""
    text = str(value or "").strip() or "Not provided"
    lines = textwrap.wrap(
        text,
        width=width - len(label),
        break_long_words=False,
        break_on_hyphens=False,
    ) or ["Not provided"]
    print(f"  {label}{lines[0]}")
    continuation = " " * (len(label) + 2)
    for line in lines[1:]:
        print(f"{continuation}{line}")


def print_questions(questions: list[dict]) -> None:
    """Pretty-print accepted questions to stdout."""
    if not questions:
        print("\nINTERVIEW LENS\nNo questions were generated.\n")
        return

    print("\n" + "=" * 78)
    print("  INTERVIEW LENS | DEMO RESULTS")
    print(f"  {len(questions)} interview question(s) generated and quality-checked")
    print("=" * 78)

    for i, q in enumerate(questions, start=1):
        print(f"\n[{i:02d}] {q.get('target_skill', 'Untitled question')}")
        print("-" * 78)
        _print_wrapped("Type:       ", q.get("question_type", "N/A").upper())
        _print_wrapped("Category:   ", q.get("category", "N/A"))
        print()
        _print_wrapped("Question:   ", q.get("question", ""))
        print()
        _print_wrapped("Why they asked: ", q.get("relevance", ""))
        rc = q.get("resume_connection")
        if rc:
            _print_wrapped("Resume link: ", rc)
        _print_wrapped("Talk like:  ", q.get("answer_hint", ""))

    print("\n" + "=" * 78)
    print("  End of demo results")
    print("=" * 78 + "\n")


def save_questions(questions: list[dict], output_path: str) -> None:
    """Save questions to a JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(questions),
        "questions": questions,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[INFO] Questions saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override max_iterations if provided via CLI
    if args.max_iterations is not None:
        config.setdefault("agent", {})["max_iterations"] = args.max_iterations

    # Setup structured logger
    logger = setup_logger(config)
    logger.info("InterviewLens starting", extra={"config_path": args.config})

    # Load inputs
    jd_text = load_text(args.jd, "Job Description")
    resume_text = load_text(args.resume, "Resume")

    logger.info(
        "Inputs loaded",
        extra={"jd_path": args.jd, "resume_path": args.resume},
    )

    # Initialise components
    gemini_client = GeminiClient(config)
    memory = SemanticMemory(gemini_client=gemini_client, config=config)
    loop = AgentLoop(gemini_client=gemini_client, memory=memory, config=config)

    # Run the agentic loop
    questions = loop.run(jd=jd_text, resume=resume_text)

    logger.info(
        "Run complete",
        extra={
            "questions_generated": len(questions),
            "total_tokens": gemini_client.total_tokens_used,
        },
    )

    # Output results
    print_questions(questions)

    if args.output:
        save_questions(questions, args.output)


if __name__ == "__main__":
    main()
