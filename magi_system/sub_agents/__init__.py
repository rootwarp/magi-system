"""Sub-agents module for MAGI system."""

from . import gemini_agent
from . import grok_agent
from . import openai_agent
from . import orchestrator_agent
from . import synthesizer_agent

__all__ = [
    "gemini_agent",
    "grok_agent",
    "openai_agent",
    "orchestrator_agent",
    "synthesizer_agent",
]
