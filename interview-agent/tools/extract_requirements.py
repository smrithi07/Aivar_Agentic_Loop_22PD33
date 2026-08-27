"""
Tool 1 — extract_requirements

Converts an unstructured job description into a structured dict with:
  required_skills, preferred_skills, responsibilities, seniority

Uses Gemini with a structured JSON prompt.  Result is cached in AgentState
so Gemini is called only once per run.
"""

import json
import logging
import re

from harness.fallback import (
    extract_json_from_text,
    handle_extract_requirements_failure,
)

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """
You are a technical recruiter assistant. Analyse the job description below and
extract structured requirements.

Return ONLY a valid JSON object — no prose, no markdown fences — with exactly
these keys:

{{
  "required_skills": ["list of must-have skills, tools, or technologies"],
  "preferred_skills": ["list of nice-to-have or bonus skills"],
  "responsibilities": ["list of key job responsibilities"],
  "seniority": "Junior | Mid | Senior | Staff | Unknown"
}}

Rules:
- Each skill must be a concise, atomic noun phrase representing an actual
  competency, technology, technical concept, qualification, or measurable
  requirement.
- Do NOT include section headings such as "About the Role", "Requirements",
  "Responsibilities", "Qualifications", or similar labels.
- Do NOT include generic job-description sentences such as "We are looking
  for a Senior Backend Engineer..." as skills.
- Do NOT copy full sentences from the JD into required_skills or
  preferred_skills.
- Do NOT include years-of-experience statements such as "5+ years of
  professional software engineering experience" or "3-5 years in backend
  development". These are hiring thresholds, not assessable competencies.
  Instead, infer seniority from them and place it in the "seniority" field.
- Each skill must be something that can independently be evaluated as a
  candidate competency.
- Split combined requirements into separate meaningful skills where
  appropriate. For example, "Python proficiency including async patterns
  and type hints" should become "Python", "Async programming", and
  "Python type hints".
- Do NOT include job titles, company descriptions, team names, or general
  introductory text as skills.
- Do NOT duplicate items across required_skills and preferred_skills.
- Responsibilities should be action-oriented sentences (e.g. "Design
  microservices").
- Seniority is inferred from titles and years-of-experience language.
Job Description:
---
{jd_text}
---
""".strip()


def extract_requirements(jd_text: str, gemini_client) -> dict:
    """
    Parse *jd_text* into a structured requirements dict.

    Parameters
    ----------
    jd_text : str
        Raw job description text.
    gemini_client : GeminiClient
        Configured Gemini client (handles retry internally).

    Returns
    -------
    dict matching EXTRACT_REQUIREMENTS_RETURN schema.
    Falls back to handle_extract_requirements_failure() on any error.
    """
    if not jd_text or not jd_text.strip():
        logger.warning("extract_requirements called with empty jd_text")
        return handle_extract_requirements_failure(jd_text or "")

    prompt = _PROMPT_TEMPLATE.format(jd_text=jd_text)

    try:
        raw = gemini_client.generate(prompt)
        parsed = extract_json_from_text(raw)

        if parsed is None:
            raise ValueError("JSON extraction returned None")

        # Normalise keys — ensure all expected fields exist
        result = {
            "required_skills": _sanitize_skills(_as_list(parsed.get("required_skills", []))),
            "preferred_skills": _sanitize_skills(_as_list(parsed.get("preferred_skills", []))),
            "responsibilities": _as_list(parsed.get("responsibilities", [])),
            "seniority": str(parsed.get("seniority", "Unknown")),
        }

        logger.info(
            "extract_requirements succeeded",
            extra={
                "n_required": len(result["required_skills"]),
                "n_preferred": len(result["preferred_skills"]),
                "n_responsibilities": len(result["responsibilities"]),
                "seniority": result["seniority"],
            },
        )
        return result

    except Exception as exc:  # noqa: BLE001
        logger.error("extract_requirements failed: %s", exc)
        return handle_extract_requirements_failure(jd_text)


def _as_list(value) -> list:
    """Coerce a value to a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []


# ---------------------------------------------------------------------------
# Post-processing filter
# ---------------------------------------------------------------------------

# Section headings that Gemini sometimes copies verbatim
_JUNK_HEADINGS = {
    "about the role", "requirements", "responsibilities", "qualifications",
    "preferred", "nice to have", "key responsibilities", "compensation",
    "about us", "who we are", "what you'll do", "what we're looking for",
    "required", "skills", "experience", "benefits", "the role",
}

# Sentence-opener patterns that indicate a copied JD sentence, not a skill
_SENTENCE_STARTERS = re.compile(
    r"^(we are|you will|you['’]ll|the team|our team|this role|the candidate|"
    r"looking for|join our|as a|in this role|the ideal|incumbent)",
    re.IGNORECASE,
)

# Years-of-experience patterns — these are hiring thresholds, not skills
# e.g. "5+ years of...", "3-5 years experience", "Minimum 2 years in..."
_YEARS_EXPERIENCE = re.compile(
    r"^(\d+\+?\s*[-–]?\s*\d*\s*years?|minimum\s+\d+|at\s+least\s+\d+|\d+\s*to\s*\d+\s*years?)",
    re.IGNORECASE,
)


def _sanitize_skills(skills: list) -> list:
    """
    Remove junk entries from a skills list:
      - Section headings ("About the Role", "Requirements", …)
      - Full JD sentences ("We are looking for…", "You will own…")
      - Job titles or role descriptions (contain em-dash or en-dash between words)
      - Entries that are too long to be a skill (> 10 words)
      - Entries that end with a period (copied sentences)
      - Blank entries
    """
    cleaned = []
    for skill in skills:
        s = skill.strip()
        if not s:
            continue
        # Too long to be a skill phrase
        if len(s.split()) > 10:
            logger.debug("_sanitize_skills: dropped (too long): %r", s)
            continue
        # Section heading
        if s.lower().rstrip(":") in _JUNK_HEADINGS:
            logger.debug("_sanitize_skills: dropped (heading): %r", s)
            continue
        # Copied full sentence opener
        if _SENTENCE_STARTERS.match(s):
            logger.debug("_sanitize_skills: dropped (sentence starter): %r", s)
            continue
        # Years-of-experience threshold (not a skill)
        if _YEARS_EXPERIENCE.match(s):
            logger.debug("_sanitize_skills: dropped (years-of-experience): %r", s)
            continue
        # Ends with period → likely a full sentence fragment
        if s.endswith("."):
            logger.debug("_sanitize_skills: dropped (ends with period): %r", s)
            continue
        cleaned.append(s)
    return cleaned
