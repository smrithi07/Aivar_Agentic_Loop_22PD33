"""
Gemini native function-calling schemas for both InterviewLens tools.

These dicts are passed directly to the Gemini API as function declarations.
They are also used as documentation of the tool contracts throughout the
codebase.
"""

EXTRACT_REQUIREMENTS_SCHEMA: dict = {
    "name": "extract_requirements",
    "description": (
        "Parse a raw job description text into structured skill categories: "
        "required_skills, preferred_skills, responsibilities, and seniority level."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "jd_text": {
                "type": "string",
                "description": "The full raw text of the job description.",
            }
        },
        "required": ["jd_text"],
    },
}

FIND_RESUME_EVIDENCE_SCHEMA: dict = {
    "name": "find_resume_evidence",
    "description": (
        "Given a specific JD requirement, find the most relevant evidence "
        "from the candidate's resume using semantic similarity search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "requirement": {
                "type": "string",
                "description": "The specific JD requirement to search evidence for.",
            },
            "resume_text": {
                "type": "string",
                "description": "The full resume text of the candidate.",
            },
        },
        "required": ["requirement", "resume_text"],
    },
}

# ---------------------------------------------------------------------------
# Return-value schemas (for documentation; not enforced at runtime)
# ---------------------------------------------------------------------------

EXTRACT_REQUIREMENTS_RETURN = {
    "required_skills": ["list of required skill strings"],
    "preferred_skills": ["list of preferred/nice-to-have skill strings"],
    "responsibilities": ["list of key responsibility strings"],
    "seniority": "Junior | Mid | Senior | Staff | Unknown",
}

FIND_RESUME_EVIDENCE_RETURN = {
    "evidence": [
        {
            "section": "Experience | Skills | Education | Projects",
            "excerpt": "verbatim or close-paraphrase excerpt from the resume",
            "relevance_score": 0.0,
        }
    ],
    "has_evidence": True,
}
