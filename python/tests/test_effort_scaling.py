"""Tests for MAGI-037: Effort scaling based on query complexity.

Tests that the planner's COMPLEXITY classification drives search depth
by capping sub-questions and searches per agent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magi_system.config import ModelConfig, PipelineConfig
from magi_system.search.effort_config import EffortLevel, EFFORT_LEVELS, get_effort_level


# ---------------------------------------------------------------------------
# EffortLevel dataclass
# ---------------------------------------------------------------------------


class TestEffortLevel:
    """Tests for the EffortLevel dataclass."""

    def test_effort_level_has_max_sub_questions(self) -> None:
        level = EffortLevel(max_sub_questions=5, max_searches_per_agent=2)
        assert level.max_sub_questions == 5

    def test_effort_level_has_max_searches_per_agent(self) -> None:
        level = EffortLevel(max_sub_questions=5, max_searches_per_agent=2)
        assert level.max_searches_per_agent == 2


# ---------------------------------------------------------------------------
# EFFORT_LEVELS constants
# ---------------------------------------------------------------------------


class TestEffortLevels:
    """Tests for the EFFORT_LEVELS mapping."""

    def test_simple_max_sub_questions(self) -> None:
        assert EFFORT_LEVELS["simple"].max_sub_questions == 5

    def test_simple_max_searches_per_agent(self) -> None:
        assert EFFORT_LEVELS["simple"].max_searches_per_agent == 2

    def test_medium_max_sub_questions(self) -> None:
        assert EFFORT_LEVELS["medium"].max_sub_questions == 8

    def test_medium_max_searches_per_agent(self) -> None:
        assert EFFORT_LEVELS["medium"].max_searches_per_agent == 3

    def test_deep_max_sub_questions(self) -> None:
        assert EFFORT_LEVELS["deep"].max_sub_questions == 15

    def test_deep_max_searches_per_agent(self) -> None:
        assert EFFORT_LEVELS["deep"].max_searches_per_agent == 5

    def test_all_three_levels_present(self) -> None:
        assert set(EFFORT_LEVELS.keys()) == {"simple", "medium", "deep"}


# ---------------------------------------------------------------------------
# get_effort_level parsing
# ---------------------------------------------------------------------------


class TestGetEffortLevel:
    """Tests for parsing COMPLEXITY from research plan text."""

    def test_parses_simple(self) -> None:
        plan = "COMPLEXITY: simple\n\nSUB_QUESTIONS:\nSQ1: What is X?"
        result = get_effort_level(plan)
        assert result == EFFORT_LEVELS["simple"]

    def test_parses_medium(self) -> None:
        plan = "COMPLEXITY: medium\n\nSUB_QUESTIONS:\nSQ1: What is X?"
        result = get_effort_level(plan)
        assert result == EFFORT_LEVELS["medium"]

    def test_parses_deep(self) -> None:
        plan = "COMPLEXITY: deep\n\nSUB_QUESTIONS:\nSQ1: What is X?"
        result = get_effort_level(plan)
        assert result == EFFORT_LEVELS["deep"]

    def test_case_insensitive_complexity_keyword(self) -> None:
        plan = "complexity: deep\n\nSQ1: What is X?"
        result = get_effort_level(plan)
        assert result == EFFORT_LEVELS["deep"]

    def test_case_insensitive_level_name(self) -> None:
        plan = "COMPLEXITY: Deep\n\nSQ1: What is X?"
        result = get_effort_level(plan)
        assert result == EFFORT_LEVELS["deep"]

    def test_mixed_case(self) -> None:
        plan = "Complexity: SIMPLE\n\nSQ1: What is X?"
        result = get_effort_level(plan)
        assert result == EFFORT_LEVELS["simple"]

    def test_no_space_after_colon(self) -> None:
        plan = "COMPLEXITY:medium\n\nSQ1: What is X?"
        result = get_effort_level(plan)
        assert result == EFFORT_LEVELS["medium"]

    def test_defaults_to_medium_for_unknown(self) -> None:
        plan = "COMPLEXITY: extreme\n\nSQ1: What is X?"
        result = get_effort_level(plan)
        assert result == EFFORT_LEVELS["medium"]

    def test_defaults_to_medium_when_no_complexity(self) -> None:
        plan = "SUB_QUESTIONS:\nSQ1: What is X?\nSQ2: What is Y?"
        result = get_effort_level(plan)
        assert result == EFFORT_LEVELS["medium"]

    def test_defaults_to_medium_for_empty_string(self) -> None:
        result = get_effort_level("")
        assert result == EFFORT_LEVELS["medium"]

    def test_complexity_embedded_in_larger_plan(self) -> None:
        plan = (
            "Some preamble text.\n"
            "COMPLEXITY: deep\n"
            "\n"
            "SUB_QUESTIONS:\n"
            "SQ1: First question?\n"
            "SQ2: Second question?\n"
        )
        result = get_effort_level(plan)
        assert result == EFFORT_LEVELS["deep"]


# ---------------------------------------------------------------------------
# DynamicSearchFanout effort scaling integration
# ---------------------------------------------------------------------------


class TestSearchFanoutEffortScaling:
    """Tests that DynamicSearchFanout uses effort scaling to cap sub-questions."""

    def _make_config(self, max_sub_questions: int = 15) -> PipelineConfig:
        return PipelineConfig(
            orchestrator=ModelConfig(name="test", model="test-model", role="orchestrator"),
            synthesis=ModelConfig(name="test", model="test-model", role="synthesis"),
            search_models=[
                ModelConfig(name="gemini_pro", model="gemini-3-pro-preview", role="search"),
            ],
            reflection_models=[],
            max_sub_questions=max_sub_questions,
        )

    def _make_plan(self, complexity: str, num_questions: int) -> str:
        lines = [f"COMPLEXITY: {complexity}\n\nSUB_QUESTIONS:"]
        for i in range(1, num_questions + 1):
            lines.append(f"SQ{i}: Question {i}?")
            lines.append(f"PRIORITY: high")
            lines.append(f"DEPENDS_ON: none")
            lines.append("")
        return "\n".join(lines)

    @pytest.mark.asyncio
    async def test_simple_caps_at_5_sub_questions(self) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        plan = self._make_plan("simple", 10)
        config = self._make_config()
        fanout = DynamicSearchFanout(name="test_fanout", config=config)

        ctx = MagicMock()
        ctx.session.state = {"research_plan": plan}

        async def fake_run_async(c):
            return
            yield  # make it an async generator

        with patch(
            "magi_system.search.search_fanout.ParallelAgent"
        ) as mock_parallel:
            mock_instance = MagicMock()
            mock_instance.run_async = fake_run_async
            mock_parallel.return_value = mock_instance

            async for _ in fanout._run_async_impl(ctx):
                pass

            # The outer ParallelAgent should receive at most 5 sub-question groups
            outer_call = mock_parallel.call_args_list[-1]
            sub_agents = outer_call.kwargs.get("sub_agents", [])
            assert len(sub_agents) <= 5

    @pytest.mark.asyncio
    async def test_config_cap_overrides_effort_when_lower(self) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        plan = self._make_plan("deep", 15)
        config = self._make_config(max_sub_questions=3)
        fanout = DynamicSearchFanout(name="test_fanout", config=config)

        ctx = MagicMock()
        ctx.session.state = {"research_plan": plan}

        async def fake_run_async(c):
            return
            yield  # make it an async generator

        with patch(
            "magi_system.search.search_fanout.ParallelAgent"
        ) as mock_parallel:
            mock_instance = MagicMock()
            mock_instance.run_async = fake_run_async
            mock_parallel.return_value = mock_instance

            async for _ in fanout._run_async_impl(ctx):
                pass

            outer_call = mock_parallel.call_args_list[-1]
            sub_agents = outer_call.kwargs.get("sub_agents", [])
            assert len(sub_agents) <= 3
