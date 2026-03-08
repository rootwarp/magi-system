"""MAGI-007: Verify existing code paths are preserved after module restructure.

Tests that all original imports continue to work and that new module directories
(clarification, consensus, planning, reflection, search, synthesis) don't
interfere with existing code.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


def _mock_google_adk():
    """Set up mocks for google.adk modules so imports resolve without credentials.

    Returns a dict of module name -> mock for cleanup.
    """
    adk_modules = {
        "google": MagicMock(),
        "google.adk": MagicMock(),
        "google.adk.agents": MagicMock(),
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
    """Mock google.adk for all tests in this module, then clean up."""
    saved = _mock_google_adk()
    # Clear cached magi_system modules so they re-import with mocks
    magi_keys = [k for k in sys.modules if k.startswith("magi_system")]
    saved_magi = {k: sys.modules.pop(k) for k in magi_keys}
    yield
    # Restore everything
    for k in list(sys.modules):
        if k.startswith("magi_system"):
            del sys.modules[k]
    sys.modules.update(saved_magi)
    _restore_modules(saved)


class TestRootAgentImport:
    """Verify root_agent is importable from the top-level package and agent module."""

    def test_import_from_package(self):
        from magi_system import root_agent

        assert root_agent is not None

    def test_import_from_agent_module(self):
        from magi_system.agent import root_agent

        assert root_agent is not None

    def test_both_refer_to_same_object(self):
        from magi_system import root_agent as ra1
        from magi_system.agent import root_agent as ra2

        assert ra1 is ra2


class TestSubAgentImports:
    """Verify all sub-agent modules are importable."""

    def test_import_gemini_agent(self):
        from magi_system.sub_agents import gemini_agent

        assert gemini_agent is not None

    def test_import_grok_agent(self):
        from magi_system.sub_agents import grok_agent

        assert grok_agent is not None

    def test_import_openai_agent(self):
        from magi_system.sub_agents import openai_agent

        assert openai_agent is not None

    def test_import_orchestrator_agent(self):
        from magi_system.sub_agents import orchestrator_agent

        assert orchestrator_agent is not None

    def test_import_synthesizer_agent(self):
        from magi_system.sub_agents import synthesizer_agent

        assert synthesizer_agent is not None


class TestDiscussionImports:
    """Verify discussion module imports work."""

    def test_import_loop_agent(self):
        from magi_system.discussion import loop_agent

        assert loop_agent is not None

    def test_import_discussion_round(self):
        from magi_system.discussion import discussion_round

        assert discussion_round is not None


class TestToolImports:
    """Verify tools module imports work (no ADK mocking needed for these)."""

    def test_import_chain_of_thought(self):
        from magi_system.tools.reasoning_tools import chain_of_thought

        assert callable(chain_of_thought)

    def test_import_compare_options(self):
        from magi_system.tools.reasoning_tools import compare_options

        assert callable(compare_options)

    def test_import_extract_page_content(self):
        from magi_system.tools.content_extraction import extract_page_content

        assert callable(extract_page_content)

    def test_import_from_tools_init(self):
        from magi_system.tools import (
            analyze_code,
            chain_of_thought,
            compare_options,
            extract_page_content,
            generate_code,
        )

        assert callable(chain_of_thought)
        assert callable(compare_options)
        assert callable(extract_page_content)
        assert callable(analyze_code)
        assert callable(generate_code)


class TestNewModulesNoInterference:
    """Verify new empty module directories don't break existing imports."""

    NEW_MODULES = [
        "magi_system.clarification",
        "magi_system.consensus",
        "magi_system.planning",
        "magi_system.reflection",
        "magi_system.search",
        "magi_system.synthesis",
    ]

    @pytest.mark.parametrize("module_name", NEW_MODULES)
    def test_new_module_importable(self, module_name):
        """Each new module directory should be importable."""
        mod = importlib.import_module(module_name)
        assert mod is not None

    def test_existing_imports_after_new_modules(self):
        """Importing new modules first should not break existing imports."""
        for name in self.NEW_MODULES:
            importlib.import_module(name)

        # Existing imports should still work
        from magi_system import root_agent
        from magi_system.tools.reasoning_tools import chain_of_thought

        assert root_agent is not None
        assert callable(chain_of_thought)
