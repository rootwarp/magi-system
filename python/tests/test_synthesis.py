"""Tests for synthesis module-level behavior, exports, and cross-module data flow (MAGI-029).

These tests complement test_synthesizer_agent.py by focusing on:
- Module __all__ exports
- Agent name verification
- Cross-module data flow (reads consensus_result from consensus, research_plan from planner)
- Instruction comprehensiveness (report sections)
"""

import sys
from unittest.mock import MagicMock

import pytest


def _mock_google_adk():
    class FakeLlmAgent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeLiteLlm:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    mock_agents_module = MagicMock()
    mock_agents_module.LlmAgent = FakeLlmAgent

    mock_lite_llm_module = MagicMock()
    mock_lite_llm_module.LiteLlm = FakeLiteLlm

    adk_modules = {
        "google": MagicMock(),
        "google.adk": MagicMock(),
        "google.adk.agents": mock_agents_module,
        "google.adk.models": MagicMock(),
        "google.adk.models.lite_llm": mock_lite_llm_module,
        "google.adk.tools": MagicMock(),
        "google.adk.tools.exit_loop_tool": MagicMock(),
        "google.adk.tools.google_search": MagicMock(),
    }
    saved = {}
    for name, mock in adk_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock
    return saved


def _restore_modules(saved):
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture(autouse=True)
def mock_adk():
    saved = _mock_google_adk()
    magi_keys = [k for k in sys.modules if k.startswith("magi_system")]
    saved_magi = {k: sys.modules.pop(k) for k in magi_keys}
    yield
    for k in list(sys.modules):
        if k.startswith("magi_system"):
            del sys.modules[k]
    sys.modules.update(saved_magi)
    _restore_modules(saved)


def _get_agent():
    from magi_system.synthesis.synthesizer_agent import synthesizer_agent

    return synthesizer_agent


# ---------------------------------------------------------------------------
# Tests: Module exports
# ---------------------------------------------------------------------------

class TestSynthesisModuleExports:
    """Verify the synthesis module __all__ and public surface."""

    def test_all_contains_synthesizer_agent(self):
        import magi_system.synthesis as mod

        assert "synthesizer_agent" in mod.__all__

    def test_all_has_one_export(self):
        import magi_system.synthesis as mod

        assert len(mod.__all__) == 1

    def test_module_docstring_exists(self):
        import magi_system.synthesis as mod

        assert mod.__doc__ is not None


# ---------------------------------------------------------------------------
# Tests: Agent identity
# ---------------------------------------------------------------------------

class TestSynthesizerAgentIdentity:
    """Verify synthesizer_agent name and output_key."""

    def test_name_is_synthesizer(self):
        agent = _get_agent()
        assert agent.name == "synthesizer"

    def test_output_key_is_final_report(self):
        agent = _get_agent()
        assert agent.output_key == "final_report"


# ---------------------------------------------------------------------------
# Tests: Cross-module data flow
# ---------------------------------------------------------------------------

class TestSynthesizerCrossModuleDataFlow:
    """Verify synthesizer reads state variables written by upstream modules."""

    def test_reads_consensus_result_from_state(self):
        """consensus_result is set by ConsensusAgent; synthesizer must reference it."""
        agent = _get_agent()
        assert "{consensus_result}" in agent.instruction

    def test_reads_research_plan_from_state(self):
        """research_plan is set by the planner; synthesizer must reference it."""
        agent = _get_agent()
        assert "{research_plan}" in agent.instruction

    def test_output_key_differs_from_consensus_result(self):
        """Synthesizer output_key should not collide with consensus_result."""
        agent = _get_agent()
        assert agent.output_key != "consensus_result"

    def test_output_key_differs_from_research_plan(self):
        """Synthesizer output_key should not collide with research_plan."""
        agent = _get_agent()
        assert agent.output_key != "research_plan"


# ---------------------------------------------------------------------------
# Tests: Instruction comprehensiveness
# ---------------------------------------------------------------------------

class TestSynthesizerInstructionComprehensiveness:
    """Verify the instruction covers all required report sections."""

    def _get_instruction(self):
        return _get_agent().instruction

    def test_mentions_confidence_levels(self):
        instruction = self._get_instruction()
        assert "HIGH CONFIDENCE" in instruction
        assert "MEDIUM CONFIDENCE" in instruction
        assert "LOW CONFIDENCE" in instruction

    def test_mentions_divergence_notes_section(self):
        instruction = self._get_instruction()
        assert "divergence notes" in instruction.lower()

    def test_mentions_confidence_summary_table(self):
        instruction = self._get_instruction()
        assert "confidence summary" in instruction.lower() or "| Finding |" in instruction

    def test_mentions_references_section(self):
        instruction = self._get_instruction()
        assert "references" in instruction.lower()

    def test_instruction_mentions_objectivity(self):
        """Should instruct the agent to be evidence-based."""
        instruction = self._get_instruction()
        assert "objective" in instruction.lower() or "evidence-based" in instruction.lower()

    def test_instruction_mentions_competing_viewpoints(self):
        """DISPUTED items should present competing viewpoints."""
        instruction = self._get_instruction()
        assert "viewpoint" in instruction.lower() or "position" in instruction.lower()
