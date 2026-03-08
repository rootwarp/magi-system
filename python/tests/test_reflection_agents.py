"""Tests for the dual reflection agents (MAGI-020)."""

from google.adk.agents import LlmAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm

from magi_system.reflection.reflection_agents import (
    REFLECTION_INSTRUCTION,
    create_dual_reflection,
    dual_reflection,
)

# Access sub-agents through the dual_reflection instance
reflection_gemini = dual_reflection.sub_agents[0]
reflection_claude = dual_reflection.sub_agents[1]


class TestDualReflectionAgent:
    """Tests for the dual_reflection ParallelAgent."""

    def test_dual_reflection_is_parallel_agent(self):
        assert isinstance(dual_reflection, ParallelAgent)

    def test_dual_reflection_has_two_sub_agents(self):
        assert len(dual_reflection.sub_agents) == 2

    def test_dual_reflection_description(self):
        assert dual_reflection.description


class TestReflectionGemini:
    """Tests for the Gemini reflection agent."""

    def test_is_llm_agent(self):
        assert isinstance(reflection_gemini, LlmAgent)

    def test_output_key(self):
        assert reflection_gemini.output_key == "reflection_gemini"

    def test_model_is_gemini(self):
        assert reflection_gemini.model == "gemini-3-pro-preview"

    def test_has_description(self):
        assert reflection_gemini.description


class TestReflectionClaude:
    """Tests for the Claude reflection agent."""

    def test_is_llm_agent(self):
        assert isinstance(reflection_claude, LlmAgent)

    def test_output_key(self):
        assert reflection_claude.output_key == "reflection_claude"

    def test_model_is_litellm(self):
        assert isinstance(reflection_claude.model, LiteLlm)

    def test_has_description(self):
        assert reflection_claude.description


class TestReflectionInstruction:
    """Tests for the shared REFLECTION_INSTRUCTION template."""

    def test_contains_research_plan_placeholder(self):
        assert "{research_plan}" in REFLECTION_INSTRUCTION

    def test_contains_all_search_results_placeholder(self):
        assert "{all_search_results}" in REFLECTION_INSTRUCTION

    def test_mentions_complete_partial_missing(self):
        assert "COMPLETE" in REFLECTION_INSTRUCTION
        assert "PARTIAL" in REFLECTION_INSTRUCTION
        assert "MISSING" in REFLECTION_INSTRUCTION

    def test_mentions_loop_back(self):
        assert "LOOP_BACK" in REFLECTION_INSTRUCTION

    def test_mentions_quality_score(self):
        assert "QUALITY_SCORE" in REFLECTION_INSTRUCTION

    def test_mentions_gap_severity_levels(self):
        assert "CRITICAL" in REFLECTION_INSTRUCTION
        assert "IMPORTANT" in REFLECTION_INSTRUCTION
        assert "MINOR" in REFLECTION_INSTRUCTION

    def test_mentions_contradiction_detection(self):
        assert "contradiction" in REFLECTION_INSTRUCTION.lower()


class TestReflectionImports:
    """Test that reflection agents are importable from the package."""

    def test_importable_from_reflection_package(self):
        from magi_system.reflection import dual_reflection as imported

        assert imported is dual_reflection

    def test_factory_creates_fresh_instances(self):
        a = create_dual_reflection()
        b = create_dual_reflection()
        assert a is not b
