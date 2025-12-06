"""MAGI System - Multi-LLM Agent System.

A multi-agent system built with Google ADK that orchestrates
Gemini, Grok, and OpenAI agents for different specialized tasks.
"""

from .agent import root_agent

__all__ = ["root_agent"]
