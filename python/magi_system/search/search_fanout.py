"""Search fan-out module for the MAGI system.

Handles decomposing research plans into sub-questions and dispatching
them to multiple search agents in parallel.
"""

import re


def _parse_sub_questions(plan: str) -> list[str]:
    """Extract sub-questions from a planner's text-based output.

    Parses lines matching SQ{N} patterns (e.g., "SQ1: ...", "SQ 1: ...",
    "SQ1 - ...") and returns the question text. Metadata lines like
    PRIORITY and DEPENDS_ON are ignored.

    Args:
        plan: Raw text output from the planner containing SQ lines.

    Returns:
        List of extracted question strings. Empty list if none found.
    """
    pattern = re.compile(r"^\s*SQ\s*\d+\s*[:\-]\s*(.+?)\s*$")
    questions: list[str] = []
    for line in plan.splitlines():
        match = pattern.match(line)
        if match:
            question = match.group(1).strip()
            if question:
                questions.append(question)
    return questions
