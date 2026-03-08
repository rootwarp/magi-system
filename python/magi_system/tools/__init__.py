"""Tools module for MAGI system agents."""

from .code_tools import analyze_code, generate_code
from .reasoning_tools import chain_of_thought, compare_options

__all__ = [
    "analyze_code",
    "generate_code",
    "chain_of_thought",
    "compare_options",
]
