"""Effort scaling configuration based on query complexity.

MAGI-037: Maps the planner's COMPLEXITY classification to concrete
limits on sub-questions and searches per agent, so that simple queries
run faster and deep queries get more thorough coverage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EffortLevel:
    """Configuration for a complexity level."""

    max_sub_questions: int
    max_searches_per_agent: int


EFFORT_LEVELS = {
    "simple": EffortLevel(max_sub_questions=5, max_searches_per_agent=2),
    "medium": EffortLevel(max_sub_questions=8, max_searches_per_agent=3),
    "deep": EffortLevel(max_sub_questions=15, max_searches_per_agent=5),
}


def get_effort_level(research_plan: str) -> EffortLevel:
    """Parse COMPLEXITY from research plan and return effort level.

    Args:
        research_plan: The research plan text containing COMPLEXITY: field.

    Returns:
        EffortLevel for the detected complexity.
    """
    plan_lower = research_plan.lower()
    for level_name in ("deep", "medium", "simple"):
        if (
            f"complexity: {level_name}" in plan_lower
            or f"complexity:{level_name}" in plan_lower
        ):
            return EFFORT_LEVELS[level_name]
    # Default to medium if not detected
    return EFFORT_LEVELS["medium"]
