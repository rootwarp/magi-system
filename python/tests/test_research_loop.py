"""Tests for the research loop factory (MAGI-025)."""

from google.adk.agents import LoopAgent, ParallelAgent

from magi_system.config import ModelConfig, PipelineConfig
from magi_system.consensus import ConsensusAgent
from magi_system.pipeline import create_research_loop
from magi_system.search import DynamicSearchFanout, SearchResultsAggregator


def _make_config(**overrides) -> PipelineConfig:
    """Create a minimal PipelineConfig for testing."""
    defaults = dict(
        orchestrator=ModelConfig(name="test", model="test-model", role="orchestrator"),
        synthesis=ModelConfig(name="test", model="test-model", role="synthesis"),
        search_models=[
            ModelConfig(name="gemini", model="gemini-test", role="search"),
        ],
        reflection_models=[
            ModelConfig(name="gemini", model="gemini-test", role="reflection"),
        ],
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class TestCreateResearchLoop:
    """Tests for create_research_loop factory function."""

    def test_returns_loop_agent(self):
        config = _make_config()
        loop = create_research_loop(config)
        assert isinstance(loop, LoopAgent)

    def test_max_iterations_from_config_default(self):
        config = _make_config()
        loop = create_research_loop(config)
        assert loop.max_iterations == 3

    def test_max_iterations_from_config_custom(self):
        config = _make_config(max_research_iterations=5)
        loop = create_research_loop(config)
        assert loop.max_iterations == 5

    def test_has_four_sub_agents(self):
        config = _make_config()
        loop = create_research_loop(config)
        assert len(loop.sub_agents) == 4

    def test_sub_agents_correct_order(self):
        config = _make_config()
        loop = create_research_loop(config)
        names = [a.name for a in loop.sub_agents]
        assert names == [
            "search_fanout",
            "search_aggregator",
            "dual_reflection",
            "consensus",
        ]

    def test_sub_agent_types(self):
        config = _make_config()
        loop = create_research_loop(config)
        agents = loop.sub_agents
        assert isinstance(agents[0], DynamicSearchFanout)
        assert isinstance(agents[1], SearchResultsAggregator)
        assert isinstance(agents[2], ParallelAgent)
        assert isinstance(agents[3], ConsensusAgent)

    def test_dual_reflection_is_fresh_instance(self):
        config = _make_config()
        loop_a = create_research_loop(config)
        loop_b = create_research_loop(config)
        assert loop_a.sub_agents[2] is not loop_b.sub_agents[2]

    def test_loop_agent_name(self):
        config = _make_config()
        loop = create_research_loop(config)
        assert loop.name == "research_loop"
