"""Discussion module for multi-agent iterative discussions."""

from .discussion_loop import loop_agent
from .discussion_round import discussion_round

__all__ = ["loop_agent", "discussion_round"]
