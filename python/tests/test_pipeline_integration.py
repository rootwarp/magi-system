"""Tests for MAGI-026: Full 6-step pipeline assembly."""

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent

from magi_system.clarification import clarifier_agent
from magi_system.planning import planner_agent


class TestRootAgentPipeline:
    """Verify root_agent is a SequentialAgent wiring Steps 1-6."""

    def test_root_agent_is_sequential_agent(self) -> None:
        from magi_system import root_agent

        assert isinstance(root_agent, SequentialAgent)

    def test_root_agent_has_four_sub_agents(self) -> None:
        from magi_system import root_agent

        assert len(root_agent.sub_agents) == 4

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

    def test_third_sub_agent_is_research_loop(self) -> None:
        from magi_system import root_agent

        third = root_agent.sub_agents[2]
        assert third.name == "research_loop"
        assert isinstance(third, LoopAgent)

    def test_fourth_sub_agent_is_synthesizer(self) -> None:
        from magi_system import root_agent

        fourth = root_agent.sub_agents[3]
        assert fourth.name == "synthesizer"
        assert isinstance(fourth, LlmAgent)

    def test_root_agent_name(self) -> None:
        from magi_system import root_agent

        assert root_agent.name == "magi_research_pipeline"

    def test_root_agent_description_updated(self) -> None:
        from magi_system import root_agent

        assert "research loop" in root_agent.description
        assert "synthesize" in root_agent.description

    def test_import_root_agent_from_package(self) -> None:
        """Verify `from magi_system import root_agent` works."""
        import magi_system

        assert hasattr(magi_system, "root_agent")
        assert isinstance(magi_system.root_agent, SequentialAgent)


class TestSubAgentTypes:
    """MAGI-026: Verify sub-agent types match expected classes."""

    def test_clarifier_is_llm_agent(self) -> None:
        from magi_system import root_agent

        assert isinstance(root_agent.sub_agents[0], LlmAgent)

    def test_planner_is_llm_agent(self) -> None:
        from magi_system import root_agent

        assert isinstance(root_agent.sub_agents[1], LlmAgent)

    def test_research_loop_is_loop_agent(self) -> None:
        from magi_system import root_agent

        assert isinstance(root_agent.sub_agents[2], LoopAgent)

    def test_synthesizer_is_llm_agent(self) -> None:
        from magi_system import root_agent

        assert isinstance(root_agent.sub_agents[3], LlmAgent)


class TestPipelineDataFlow:
    """MAGI-026: Verify output_key chain ensures correct data flow."""

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

    def test_synthesizer_output_key_is_final_report(self) -> None:
        from magi_system import root_agent

        synthesizer = root_agent.sub_agents[3]
        assert synthesizer.output_key == "final_report"


class TestResearchLoopConfig:
    """MAGI-026: Verify research loop is properly configured."""

    def test_research_loop_has_sub_agents(self) -> None:
        from magi_system import root_agent

        loop = root_agent.sub_agents[2]
        assert len(loop.sub_agents) == 4  # fanout, aggregator, reflection, consensus

    def test_research_loop_max_iterations(self) -> None:
        from magi_system import root_agent

        loop = root_agent.sub_agents[2]
        assert loop.max_iterations > 0
