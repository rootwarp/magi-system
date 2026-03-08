"""Tests for MAGI-008: Clarification agent configuration and exports."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


def _mock_google_adk():
    """Set up mocks for google.adk modules so imports resolve without credentials."""
    # Create a real-ish LlmAgent mock that captures constructor args
    class FakeLlmAgent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    mock_agents_module = MagicMock()
    mock_agents_module.LlmAgent = FakeLlmAgent

    adk_modules = {
        "google": MagicMock(),
        "google.adk": MagicMock(),
        "google.adk.agents": mock_agents_module,
        "google.adk.models": MagicMock(),
        "google.adk.models.lite_llm": MagicMock(),
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


class TestClarifierAgentExport:
    """Verify that the clarifier agent is exported from the clarification module."""

    def test_import_clarifier_agent_from_clarification(self):
        from magi_system.clarification import clarifier_agent

        assert clarifier_agent is not None

    def test_import_clarifier_agent_module(self):
        mod = importlib.import_module(
            "magi_system.clarification.clarifier_agent"
        )
        assert hasattr(mod, "agent")


class TestClarifierAgentConfiguration:
    """Verify clarifier agent has the correct configuration."""

    def _get_agent(self):
        from magi_system.clarification.clarifier_agent import agent

        return agent

    def test_model_is_gemini_pro(self):
        agent = self._get_agent()
        assert agent.model == "gemini-3-pro-preview"

    def test_output_key_is_research_brief(self):
        agent = self._get_agent()
        assert agent.output_key == "research_brief"

    def test_name_is_set(self):
        agent = self._get_agent()
        assert agent.name is not None
        assert len(agent.name) > 0

    def test_description_is_set(self):
        agent = self._get_agent()
        assert agent.description is not None
        assert len(agent.description) > 0


class TestClarifierAgentInstruction:
    """Verify the clarifier agent instruction contains required guidance."""

    def _get_instruction(self):
        from magi_system.clarification.clarifier_agent import agent

        return agent.instruction

    def test_instruction_mentions_clarifying_questions(self):
        instruction = self._get_instruction()
        assert "clarifying question" in instruction.lower()

    def test_instruction_mentions_3_to_5_questions(self):
        instruction = self._get_instruction()
        assert "3" in instruction and "5" in instruction

    def test_instruction_mentions_research_brief(self):
        instruction = self._get_instruction()
        assert "research brief" in instruction.lower()

    def test_instruction_handles_clear_queries(self):
        """Instruction should guide agent to handle already-clear queries."""
        instruction = self._get_instruction()
        lower = instruction.lower()
        assert "clear" in lower or "straightforward" in lower

    def test_instruction_includes_brief_format(self):
        """Instruction should include format guidance for the research brief."""
        instruction = self._get_instruction()
        lower = instruction.lower()
        assert "scope" in lower
        assert "aspect" in lower or "key" in lower

    def test_instruction_includes_constraints(self):
        """Instruction should mention constraints in the brief format."""
        instruction = self._get_instruction()
        assert "constraint" in instruction.lower()
