"""Tests for reflection module-level behavior, exports, and cross-module references (MAGI-029).

These tests complement test_reflection_agents.py by focusing on:
- Module __all__ exports
- Cross-agent consistency (shared instruction, distinct output_keys)
- Sub-agent membership in the ParallelAgent
- Agent name correctness
"""

from google.adk.agents import LlmAgent, ParallelAgent

from magi_system.reflection import dual_reflection
from magi_system.reflection.reflection_agents import (
    REFLECTION_INSTRUCTION,
    reflection_claude,
    reflection_gemini,
)


class TestReflectionModuleExports:
    """Verify the reflection module __all__ and public surface."""

    def test_all_contains_dual_reflection(self):
        import magi_system.reflection as mod

        assert "dual_reflection" in mod.__all__

    def test_all_length(self):
        import magi_system.reflection as mod

        assert len(mod.__all__) == 1

    def test_module_docstring_exists(self):
        import magi_system.reflection as mod

        assert mod.__doc__ is not None


class TestDualReflectionSubAgents:
    """Verify sub-agent names and membership within dual_reflection."""

    def test_sub_agent_names(self):
        names = {a.name for a in dual_reflection.sub_agents}
        assert names == {"reflection_gemini", "reflection_claude"}

    def test_sub_agents_are_the_module_level_instances(self):
        assert dual_reflection.sub_agents[0] is reflection_gemini
        assert dual_reflection.sub_agents[1] is reflection_claude

    def test_dual_reflection_name(self):
        assert dual_reflection.name == "dual_reflection"


class TestReflectionAgentConsistency:
    """Cross-agent checks: shared instruction, distinct output keys."""

    def test_both_agents_share_same_instruction(self):
        assert reflection_gemini.instruction is reflection_claude.instruction

    def test_output_keys_are_distinct(self):
        assert reflection_gemini.output_key != reflection_claude.output_key

    def test_output_keys_match_agent_names(self):
        assert reflection_gemini.output_key == "reflection_gemini"
        assert reflection_claude.output_key == "reflection_claude"

    def test_both_reference_research_plan(self):
        assert "{research_plan}" in reflection_gemini.instruction
        assert "{research_plan}" in reflection_claude.instruction

    def test_both_reference_all_search_results(self):
        assert "{all_search_results}" in reflection_gemini.instruction
        assert "{all_search_results}" in reflection_claude.instruction

    def test_models_use_different_providers(self):
        """Gemini uses a string model ID; Claude uses LiteLlm wrapper."""
        assert isinstance(reflection_gemini.model, str)
        assert not isinstance(reflection_claude.model, str)
