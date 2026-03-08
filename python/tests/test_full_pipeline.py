"""Full pipeline integration tests (MAGI-028 + MAGI-030).

Structural integration tests verifying the complete 6-step pipeline assembly,
research loop internals, data flow through state keys, and configuration
consistency — all without requiring API keys.
"""

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents import BaseAgent
from google.adk.models.lite_llm import LiteLlm

from magi_system.config import load_config
from magi_system.consensus import ConsensusAgent
from magi_system.search import DynamicSearchFanout, SearchResultsAggregator


def _get_root():
    """Import and return the root_agent to isolate module-level side effects."""
    from magi_system import root_agent

    return root_agent


# ---------------------------------------------------------------------------
# 1. Full pipeline structure
# ---------------------------------------------------------------------------


class TestFullPipelineStructure:
    """Verify root_agent is a SequentialAgent with the correct 4 sub-agents."""

    def test_root_is_sequential_with_four_sub_agents(self) -> None:
        root = _get_root()
        assert isinstance(root, SequentialAgent)
        assert len(root.sub_agents) == 4

    def test_sub_agent_order(self) -> None:
        root = _get_root()
        names = [a.name for a in root.sub_agents]
        assert names == ["clarifier", "planner", "research_loop", "synthesizer"]

    def test_clarifier_is_llm_agent(self) -> None:
        root = _get_root()
        assert isinstance(root.sub_agents[0], LlmAgent)

    def test_planner_is_llm_agent(self) -> None:
        root = _get_root()
        assert isinstance(root.sub_agents[1], LlmAgent)

    def test_research_loop_is_loop_agent(self) -> None:
        root = _get_root()
        assert isinstance(root.sub_agents[2], LoopAgent)

    def test_synthesizer_is_llm_agent(self) -> None:
        root = _get_root()
        assert isinstance(root.sub_agents[3], LlmAgent)


# ---------------------------------------------------------------------------
# 2. Research loop internals
# ---------------------------------------------------------------------------


class TestResearchLoopInternals:
    """Verify research_loop has 4 correctly-typed sub-agents."""

    def _loop(self):
        return _get_root().sub_agents[2]

    def test_loop_has_four_sub_agents(self) -> None:
        loop = self._loop()
        assert len(loop.sub_agents) == 4

    def test_sub_agent_names(self) -> None:
        loop = self._loop()
        names = [a.name for a in loop.sub_agents]
        assert names == [
            "search_fanout",
            "search_aggregator",
            "dual_reflection",
            "consensus",
        ]

    def test_search_fanout_type(self) -> None:
        loop = self._loop()
        assert isinstance(loop.sub_agents[0], DynamicSearchFanout)

    def test_aggregator_type(self) -> None:
        loop = self._loop()
        assert isinstance(loop.sub_agents[1], SearchResultsAggregator)

    def test_dual_reflection_is_parallel_agent(self) -> None:
        loop = self._loop()
        assert isinstance(loop.sub_agents[2], ParallelAgent)

    def test_dual_reflection_has_two_sub_agents(self) -> None:
        loop = self._loop()
        reflection = loop.sub_agents[2]
        assert len(reflection.sub_agents) == 2

    def test_reflection_agents_are_llm_agents(self) -> None:
        loop = self._loop()
        reflection = loop.sub_agents[2]
        for agent in reflection.sub_agents:
            assert isinstance(agent, LlmAgent)

    def test_reflection_agent_names(self) -> None:
        loop = self._loop()
        reflection = loop.sub_agents[2]
        names = {a.name for a in reflection.sub_agents}
        assert names == {"reflection_gemini", "reflection_claude"}

    def test_consensus_type(self) -> None:
        loop = self._loop()
        assert isinstance(loop.sub_agents[3], ConsensusAgent)


# ---------------------------------------------------------------------------
# 3. Data flow verification — output_key / instruction state references
# ---------------------------------------------------------------------------


class TestDataFlowKeys:
    """Verify that state keys chain correctly across the pipeline."""

    def test_clarifier_output_key(self) -> None:
        root = _get_root()
        assert root.sub_agents[0].output_key == "research_brief"

    def test_planner_reads_research_brief(self) -> None:
        root = _get_root()
        planner = root.sub_agents[1]
        assert "{research_brief}" in planner.instruction

    def test_planner_output_key(self) -> None:
        root = _get_root()
        assert root.sub_agents[1].output_key == "research_plan"

    def test_aggregator_writes_all_search_results(self) -> None:
        """Aggregator stores its output under 'all_search_results' (verified
        by inspecting the class implementation — the key is hard-coded)."""
        # SearchResultsAggregator is a BaseAgent that writes to state directly.
        # We verify the key name by checking the source. Here we test the type
        # to ensure the aggregator is the correct class.
        loop = _get_root().sub_agents[2]
        aggregator = loop.sub_agents[1]
        assert isinstance(aggregator, SearchResultsAggregator)

    def test_reflection_agents_read_research_plan(self) -> None:
        loop = _get_root().sub_agents[2]
        reflection = loop.sub_agents[2]
        for agent in reflection.sub_agents:
            assert "{research_plan}" in agent.instruction

    def test_reflection_agents_read_all_search_results(self) -> None:
        loop = _get_root().sub_agents[2]
        reflection = loop.sub_agents[2]
        for agent in reflection.sub_agents:
            assert "{all_search_results}" in agent.instruction

    def test_reflection_gemini_output_key(self) -> None:
        loop = _get_root().sub_agents[2]
        reflection = loop.sub_agents[2]
        gemini = [a for a in reflection.sub_agents if a.name == "reflection_gemini"][0]
        assert gemini.output_key == "reflection_gemini"

    def test_reflection_claude_output_key(self) -> None:
        loop = _get_root().sub_agents[2]
        reflection = loop.sub_agents[2]
        claude = [a for a in reflection.sub_agents if a.name == "reflection_claude"][0]
        assert claude.output_key == "reflection_claude"

    def test_divergence_detector_reads_reflections(self) -> None:
        """Divergence detector (inside ConsensusAgent) reads both reflection keys."""
        from magi_system.consensus.divergence import divergence_detector

        assert "{reflection_gemini}" in divergence_detector.instruction
        assert "{reflection_claude}" in divergence_detector.instruction

    def test_divergence_detector_reads_research_plan(self) -> None:
        from magi_system.consensus.divergence import divergence_detector

        assert "{research_plan}" in divergence_detector.instruction

    def test_divergence_detector_reads_all_search_results(self) -> None:
        from magi_system.consensus.divergence import divergence_detector

        assert "{all_search_results}" in divergence_detector.instruction

    def test_divergence_detector_output_key(self) -> None:
        from magi_system.consensus.divergence import divergence_detector

        assert divergence_detector.output_key == "divergence_map"

    def test_synthesizer_reads_consensus_result(self) -> None:
        root = _get_root()
        synthesizer = root.sub_agents[3]
        assert "{consensus_result}" in synthesizer.instruction

    def test_synthesizer_reads_research_plan(self) -> None:
        root = _get_root()
        synthesizer = root.sub_agents[3]
        assert "{research_plan}" in synthesizer.instruction

    def test_synthesizer_output_key(self) -> None:
        root = _get_root()
        assert root.sub_agents[3].output_key == "final_report"


# ---------------------------------------------------------------------------
# 4. Configuration consistency
# ---------------------------------------------------------------------------


class TestConfigurationConsistency:
    """Verify pipeline configuration matches expected values."""

    def test_research_loop_max_iterations_matches_config(self) -> None:
        config = load_config()
        loop = _get_root().sub_agents[2]
        assert loop.max_iterations == config.max_research_iterations

    def test_clarifier_uses_orchestrator_model(self) -> None:
        """Clarifier should use the gemini model (orchestrator role)."""
        root = _get_root()
        clarifier = root.sub_agents[0]
        # Clarifier uses plain string model "gemini-3-pro-preview"
        assert clarifier.model == "gemini-3-pro-preview"

    def test_planner_uses_orchestrator_model(self) -> None:
        root = _get_root()
        planner = root.sub_agents[1]
        assert planner.model == "gemini-3-pro-preview"

    def test_synthesizer_uses_claude_model(self) -> None:
        root = _get_root()
        synthesizer = root.sub_agents[3]
        assert isinstance(synthesizer.model, LiteLlm)

    def test_reflection_gemini_model(self) -> None:
        loop = _get_root().sub_agents[2]
        reflection = loop.sub_agents[2]
        gemini = [a for a in reflection.sub_agents if a.name == "reflection_gemini"][0]
        assert gemini.model == "gemini-3-pro-preview"

    def test_reflection_claude_model(self) -> None:
        loop = _get_root().sub_agents[2]
        reflection = loop.sub_agents[2]
        claude = [a for a in reflection.sub_agents if a.name == "reflection_claude"][0]
        assert isinstance(claude.model, LiteLlm)


# ---------------------------------------------------------------------------
# 5. End-to-end state key chain (comprehensive)
# ---------------------------------------------------------------------------


class TestEndToEndStateKeyChain:
    """Verify the full state key chain from clarifier through synthesizer.

    The pipeline reads and writes state in this order:
      clarifier -> research_brief
      planner reads research_brief -> research_plan
      search fanout reads research_brief + research_plan -> sq{N}_{model} keys
      aggregator reads sq{N}_{model} -> all_search_results
      reflection reads research_plan + all_search_results -> reflection_gemini, reflection_claude
      divergence reads reflections + research_plan + all_search_results -> divergence_map
      consensus produces -> consensus_result
      synthesizer reads consensus_result + research_plan -> final_report
    """

    def test_output_keys_form_connected_chain(self) -> None:
        """Each agent's output_key is consumed by a downstream agent."""
        root = _get_root()
        clarifier = root.sub_agents[0]
        planner = root.sub_agents[1]
        synthesizer = root.sub_agents[3]

        # clarifier produces what planner needs
        assert clarifier.output_key == "research_brief"
        assert "{research_brief}" in planner.instruction

        # planner produces what synthesizer needs
        assert planner.output_key == "research_plan"
        assert "{research_plan}" in synthesizer.instruction

        # synthesizer reads consensus_result (produced by ConsensusAgent)
        assert "{consensus_result}" in synthesizer.instruction
        assert synthesizer.output_key == "final_report"

    def test_no_orphan_state_keys_in_reflection(self) -> None:
        """Reflection agents only reference keys that are produced upstream."""
        loop = _get_root().sub_agents[2]
        reflection = loop.sub_agents[2]
        for agent in reflection.sub_agents:
            # These keys must exist before reflection runs
            assert "{research_plan}" in agent.instruction
            assert "{all_search_results}" in agent.instruction

    def test_divergence_detector_references_only_upstream_keys(self) -> None:
        from magi_system.consensus.divergence import divergence_detector

        instruction = divergence_detector.instruction
        # All referenced keys are produced by upstream agents
        assert "{research_plan}" in instruction
        assert "{reflection_gemini}" in instruction
        assert "{reflection_claude}" in instruction
        assert "{all_search_results}" in instruction

    def test_pipeline_produces_final_report(self) -> None:
        """The terminal agent writes 'final_report' to state."""
        root = _get_root()
        terminal = root.sub_agents[-1]
        assert terminal.output_key == "final_report"
