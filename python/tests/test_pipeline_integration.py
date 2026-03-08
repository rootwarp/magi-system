"""Tests for MAGI-015/018: Wiring Steps 1-3 into partial pipeline."""

from google.adk.agents import LlmAgent, SequentialAgent

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


class TestSubAgentTypes:
    """MAGI-018: Verify sub-agent types match expected classes."""

    def test_clarifier_is_llm_agent(self) -> None:
        from magi_system import root_agent

        assert isinstance(root_agent.sub_agents[0], LlmAgent)

    def test_planner_is_llm_agent(self) -> None:
        from magi_system import root_agent

        assert isinstance(root_agent.sub_agents[1], LlmAgent)

    def test_search_fanout_is_dynamic_search_fanout(self) -> None:
        from magi_system import root_agent

        assert isinstance(root_agent.sub_agents[2], DynamicSearchFanout)


class TestPipelineDataFlow:
    """MAGI-018: Verify output_key chain ensures correct data flow."""

    def test_clarifier_output_key_is_research_brief(self) -> None:
        from magi_system import root_agent

        clarifier = root_agent.sub_agents[0]
        assert clarifier.output_key == "research_brief"

    def test_planner_output_key_is_research_plan(self) -> None:
        from magi_system import root_agent

        planner = root_agent.sub_agents[1]
        assert planner.output_key == "research_plan"

    def test_planner_instruction_references_research_brief(self) -> None:
        """Planner instruction must template in {research_brief} from state."""
        from magi_system import root_agent

        planner = root_agent.sub_agents[1]
        assert "{research_brief}" in planner.instruction


class TestSearchFanoutConfig:
    """MAGI-018: Verify DynamicSearchFanout has correct configuration."""

    def test_search_fanout_has_config(self) -> None:
        from magi_system import root_agent

        fanout = root_agent.sub_agents[2]
        assert hasattr(fanout, "config")

    def test_max_sub_questions_is_positive(self) -> None:
        from magi_system import root_agent

        fanout = root_agent.sub_agents[2]
        assert fanout.config.max_sub_questions > 0

    def test_search_models_count(self) -> None:
        """Pipeline should have at least 2 search models configured."""
        from magi_system import root_agent

        fanout = root_agent.sub_agents[2]
        enabled = [m for m in fanout.config.search_models if m.enabled]
        assert len(enabled) >= 2

    def test_search_models_have_search_role(self) -> None:
        from magi_system import root_agent

        fanout = root_agent.sub_agents[2]
        for model_cfg in fanout.config.search_models:
            assert model_cfg.role == "search"
