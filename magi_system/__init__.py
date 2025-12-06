"""MAGI System - Multi-LLM Agent Voting System.

A multi-agent voting system built with Google ADK that runs
Gemini, Grok, and OpenAI agents in parallel and synthesizes their responses.
"""

from .agent import root_agent

__all__ = ["root_agent"]
