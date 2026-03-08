"""Reasoning and analysis tools for the Grok agent."""

from typing import List


def chain_of_thought(problem: str) -> dict:
    """Initiate step-by-step reasoning for a problem.

    Args:
        problem: The problem statement to analyze.

    Returns:
        A dictionary with reasoning framework and instructions.
    """
    return {
        "status": "ready",
        "problem": problem,
        "framework": {
            "step_1": "Identify the key components of the problem",
            "step_2": "Break down into smaller sub-problems",
            "step_3": "Analyze each sub-problem systematically",
            "step_4": "Synthesize findings into a coherent solution",
            "step_5": "Validate the solution against original requirements",
        },
        "message": "Chain of thought reasoning initiated. "
        "Apply the framework above to analyze the problem systematically.",
    }


def compare_options(options: List[str]) -> dict:
    """Compare and evaluate multiple options.

    Args:
        options: List of options to compare.

    Returns:
        A dictionary with comparison framework and instructions.
    """
    if not options or len(options) < 2:
        return {
            "status": "error",
            "message": "At least 2 options are required for comparison.",
        }

    return {
        "status": "ready",
        "options": options,
        "option_count": len(options),
        "evaluation_criteria": [
            "Feasibility",
            "Cost/Effort",
            "Risk",
            "Potential Impact",
            "Time to Implementation",
        ],
        "message": f"Ready to compare {len(options)} options. "
        "Evaluate each option against the criteria above and provide a recommendation.",
    }
