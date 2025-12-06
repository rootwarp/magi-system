"""Code analysis and generation tools for the OpenAI agent."""


def analyze_code(code: str) -> dict:
    """Analyze code structure and provide insights.

    Args:
        code: The source code to analyze.

    Returns:
        A dictionary containing analysis results with structure info and suggestions.
    """
    lines = code.strip().split("\n")
    return {
        "status": "success",
        "analysis": {
            "line_count": len(lines),
            "has_functions": "def " in code,
            "has_classes": "class " in code,
            "has_imports": "import " in code or "from " in code,
        },
        "message": "Code analysis complete. Review the structure details above.",
    }


def generate_code(description: str, language: str = "python") -> dict:
    """Generate code based on a description.

    Args:
        description: What the code should do.
        language: The programming language to use (default: python).

    Returns:
        A dictionary containing the generation status and instructions.
    """
    return {
        "status": "ready",
        "language": language,
        "description": description,
        "message": f"Ready to generate {language} code for: {description}. "
        "Please provide the implementation based on the description.",
    }
