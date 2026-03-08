"""Tests for MAGI-009: Planning agent configuration and structure."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


def _mock_google_adk():
    """Set up mocks for google.adk modules so imports resolve without credentials."""
    mock_llm_agent = MagicMock()

    adk_modules = {
        "google": MagicMock(),
        "google.adk": MagicMock(),
        "google.adk.agents": MagicMock(LlmAgent=mock_llm_agent),
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
    return saved, mock_llm_agent


def _restore_modules(saved):
    """Restore original sys.modules entries."""
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture(autouse=True)
def mock_adk():
    """Mock google.adk for all tests in this module, then clean up."""
    saved, mock_llm_agent = _mock_google_adk()
    # Clear cached magi_system modules so they re-import with mocks
    magi_keys = [k for k in sys.modules if k.startswith("magi_system")]
    saved_magi = {k: sys.modules.pop(k) for k in magi_keys}
    yield mock_llm_agent
    # Restore everything
    for k in list(sys.modules):
        if k.startswith("magi_system"):
            del sys.modules[k]
    sys.modules.update(saved_magi)
    _restore_modules(saved)


class TestPlannerAgentImport:
    """Verify planner_agent is importable from the planning module."""

    def test_import_planner_agent_from_planning(self):
        from magi_system.planning import planner_agent

        assert planner_agent is not None

    def test_import_planner_agent_module(self):
        mod = importlib.import_module("magi_system.planning.planner_agent")
        assert mod is not None

    def test_planner_agent_has_agent_attribute(self):
        from magi_system.planning.planner_agent import planner_agent

        assert planner_agent is not None


class TestPlannerAgentConfiguration:
    """Verify planner agent is configured correctly per MAGI-009 spec."""

    def test_agent_created_with_llm_agent(self, mock_adk):
        from magi_system.planning.planner_agent import planner_agent  # noqa: F811

        # LlmAgent should have been called with name="planner"
        planner_calls = [
            c for c in mock_adk.call_args_list if c[1].get("name") == "planner"
        ]
        assert len(planner_calls) == 1

    def test_agent_name_is_planner(self, mock_adk):
        # Re-import to trigger fresh construction
        from magi_system.planning import planner_agent  # noqa: F811

        planner_calls = [
            c for c in mock_adk.call_args_list if c[1].get("name") == "planner"
        ]
        assert len(planner_calls) == 1
        assert planner_calls[0][1]["name"] == "planner"

    def test_agent_model_is_gemini_3_pro(self, mock_adk):
        from magi_system.planning import planner_agent  # noqa: F811

        planner_calls = [
            c for c in mock_adk.call_args_list if c[1].get("name") == "planner"
        ]
        assert planner_calls[0][1]["model"] == "gemini-3-pro-preview"

    def test_agent_output_key_is_research_plan(self, mock_adk):
        from magi_system.planning import planner_agent  # noqa: F811

        planner_calls = [
            c for c in mock_adk.call_args_list if c[1].get("name") == "planner"
        ]
        assert planner_calls[0][1]["output_key"] == "research_plan"

    def test_agent_has_description(self, mock_adk):
        from magi_system.planning import planner_agent  # noqa: F811

        planner_calls = [
            c for c in mock_adk.call_args_list if c[1].get("name") == "planner"
        ]
        assert "description" in planner_calls[0][1]
        assert len(planner_calls[0][1]["description"]) > 0


def _get_planner_kwargs(mock_adk):
    """Helper to find the planner LlmAgent call kwargs from call_args_list."""
    from magi_system.planning import planner_agent  # noqa: F811

    planner_calls = [
        c for c in mock_adk.call_args_list if c[1].get("name") == "planner"
    ]
    assert len(planner_calls) == 1, f"Expected 1 planner call, got {len(planner_calls)}"
    return planner_calls[0][1]


class TestPlannerAgentInstruction:
    """Verify the instruction template contains required elements."""

    def test_instruction_reads_research_brief(self, mock_adk):
        call_kwargs = _get_planner_kwargs(mock_adk)
        assert "{research_brief}" in call_kwargs["instruction"]

    def test_instruction_specifies_complexity_levels(self, mock_adk):
        instruction = _get_planner_kwargs(mock_adk)["instruction"]
        assert "simple" in instruction.lower()
        assert "medium" in instruction.lower()
        assert "deep" in instruction.lower()

    def test_instruction_specifies_sub_question_ranges(self, mock_adk):
        instruction = _get_planner_kwargs(mock_adk)["instruction"]
        assert "3-5" in instruction or ("3" in instruction and "5" in instruction)
        assert "5-8" in instruction or ("5" in instruction and "8" in instruction)
        assert "8-15" in instruction or ("8" in instruction and "15" in instruction)

    def test_instruction_has_complexity_field(self, mock_adk):
        instruction = _get_planner_kwargs(mock_adk)["instruction"]
        assert "COMPLEXITY:" in instruction

    def test_instruction_has_sub_question_format(self, mock_adk):
        instruction = _get_planner_kwargs(mock_adk)["instruction"]
        assert "SQ" in instruction

    def test_instruction_has_priority_field(self, mock_adk):
        instruction = _get_planner_kwargs(mock_adk)["instruction"]
        assert "PRIORITY:" in instruction

    def test_instruction_has_depends_on_field(self, mock_adk):
        instruction = _get_planner_kwargs(mock_adk)["instruction"]
        assert "DEPENDS_ON:" in instruction

    def test_instruction_priority_values(self, mock_adk):
        instruction = _get_planner_kwargs(mock_adk)["instruction"]
        assert "high" in instruction.lower()
        assert "medium" in instruction.lower()
        assert "low" in instruction.lower()
