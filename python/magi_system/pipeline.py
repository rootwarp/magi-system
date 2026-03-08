"""Research loop assembly for the MAGI pipeline (MAGI-025).

Factory function that wires Steps 3-5 (search fanout → aggregator →
dual reflection → consensus) into a single LoopAgent.
"""

from __future__ import annotations

from google.adk.agents import LoopAgent

from .config import PipelineConfig
from .consensus import ConsensusAgent
from .reflection import create_dual_reflection
from .search import DynamicSearchFanout, SearchResultsAggregator


def create_research_loop(config: PipelineConfig) -> LoopAgent:
    """Create the research loop (Steps 3-5).

    The loop runs: search fanout → aggregator → dual reflection → consensus.
    It iterates until max_research_iterations is reached.

    Args:
        config: Pipeline configuration with model and iteration settings.

    Returns:
        A LoopAgent wrapping the four research sub-agents.
    """
    search_fanout = DynamicSearchFanout(
        name="search_fanout",
        config=config,
    )

    aggregator = SearchResultsAggregator(
        name="search_aggregator",
    )

    consensus = ConsensusAgent(
        name="consensus",
        config=config,
    )

    return LoopAgent(
        name="research_loop",
        sub_agents=[search_fanout, aggregator, create_dual_reflection(), consensus],
        max_iterations=config.max_research_iterations,
        description="Research loop: search → aggregate → reflect → consensus.",
    )
