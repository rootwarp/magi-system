"""MAGI System - Multi-LLM Agent Voting System.

A multi-agent voting system built with Google ADK that runs
Gemini, Grok, and OpenAI agents in parallel and synthesizes their responses.
"""

from .agent import root_agent
from .logging_config import StructuredFormatter, configure_logging

__all__ = ["root_agent", "StructuredFormatter", "configure_logging"]
