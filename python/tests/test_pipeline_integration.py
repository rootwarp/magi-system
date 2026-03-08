"""Tests for MAGI-015: Wiring Steps 1-3 into partial pipeline."""

from google.adk.agents import SequentialAgent

from magi_system.clarification import clarifier_agent
from magi_system.planning import planner_agent
from magi_system.search import DynamicSearchFanout


class TestRootAgentPipeline:
    """Verify root_agent is a SequentialAgent wiring Steps 1-3."""

    def test_root_agent_is_sequential_agent(self) -> None:
        from magi_system import root_agent

        assert isinstance(root_agent, SequentialAgent)

    def test_root_agent_has_three_sub_agents(self) -> None:
        from magi_system import root_agent

        assert len(root_agent.sub_agents) == 3

    def test_first_sub_agent_is_clarifier(self) -> None:
        from magi_system import root_agent

        first = root_agent.sub_agents[0]
        assert first.name == "clarifier"
        assert first is clarifier_agent.agent

    def test_second_sub_agent_is_planner(self) -> None:
        from magi_system import root_agent

        second = root_agent.sub_agents[1]
        assert second.name == "planner"
        assert second is planner_agent

    def test_third_sub_agent_is_search_fanout(self) -> None:
        from magi_system import root_agent

        third = root_agent.sub_agents[2]
        assert third.name == "search_fanout"
        assert isinstance(third, DynamicSearchFanout)

    def test_root_agent_name(self) -> None:
        from magi_system import root_agent

        assert root_agent.name == "magi_research_pipeline"

    def test_import_root_agent_from_package(self) -> None:
        """Verify `from magi_system import root_agent` works."""
        import magi_system

        assert hasattr(magi_system, "root_agent")
        assert isinstance(magi_system.root_agent, SequentialAgent)
