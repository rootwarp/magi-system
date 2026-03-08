"""Tests for MAGI-024: Synthesis agent configuration and exports."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


def _mock_google_adk():
    """Set up mocks for google.adk modules so imports resolve without credentials."""

    class FakeLlmAgent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    mock_agents_module = MagicMock()
    mock_agents_module.LlmAgent = FakeLlmAgent

    class FakeLiteLlm:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

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
    """Restore original sys.modules entries."""
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture(autouse=True)
def mock_adk():
    """Mock google.adk for all tests, then clean up."""
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


class TestSynthesizerAgentExport:
    """Verify synthesizer_agent is importable from the synthesis module."""

    def test_import_synthesizer_agent_from_synthesis(self):
        from magi_system.synthesis import synthesizer_agent

        assert synthesizer_agent is not None

    def test_import_synthesizer_agent_module(self):
        mod = importlib.import_module(
            "magi_system.synthesis.synthesizer_agent"
        )
        assert hasattr(mod, "synthesizer_agent")


class TestSynthesizerAgentConfiguration:
    """Verify synthesizer agent is configured correctly per MAGI-024 spec."""

    def test_agent_is_llm_agent_instance(self):
        from google.adk.agents import LlmAgent

        agent = _get_agent()
        assert isinstance(agent, LlmAgent)

    def test_model_is_litellm_with_claude(self):
        agent = _get_agent()
        assert hasattr(agent.model, "model")
        assert agent.model.model == "anthropic/claude-opus-4-6"

    def test_output_key_is_final_report(self):
        agent = _get_agent()
        assert agent.output_key == "final_report"

    def test_name_is_set(self):
        agent = _get_agent()
        assert agent.name == "synthesizer"

    def test_description_is_set(self):
        agent = _get_agent()
        assert agent.description is not None
        assert len(agent.description) > 0


class TestSynthesizerAgentInstruction:
    """Verify the instruction template contains required report structure."""

    def _get_instruction(self):
        return _get_agent().instruction

    def test_instruction_contains_consensus_result_placeholder(self):
        instruction = self._get_instruction()
        assert "{consensus_result}" in instruction

    def test_instruction_contains_research_plan_placeholder(self):
        instruction = self._get_instruction()
        assert "{research_plan}" in instruction

    def test_instruction_mentions_executive_summary(self):
        instruction = self._get_instruction()
        assert "executive summary" in instruction.lower()

    def test_instruction_mentions_key_findings(self):
        instruction = self._get_instruction()
        assert "key findings" in instruction.lower() or "findings" in instruction.lower()

    def test_instruction_mentions_analysis(self):
        instruction = self._get_instruction()
        assert "analysis" in instruction.lower()

    def test_instruction_mentions_limitations(self):
        instruction = self._get_instruction()
        assert "limitation" in instruction.lower()

    def test_instruction_mentions_citation_format(self):
        instruction = self._get_instruction()
        assert "[1]" in instruction and "[2]" in instruction

    def test_instruction_mentions_disputed_handling(self):
        instruction = self._get_instruction()
        assert "[DISPUTED]" in instruction

    def test_instruction_mentions_confidence(self):
        instruction = self._get_instruction()
        assert "confidence" in instruction.lower()

    def test_instruction_mentions_references(self):
        instruction = self._get_instruction()
        assert "reference" in instruction.lower()

    def test_instruction_mentions_divergence(self):
        instruction = self._get_instruction()
        assert "divergence" in instruction.lower()
