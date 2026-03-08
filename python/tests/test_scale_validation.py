"""Tests for MAGI-014: Validate dynamic ParallelAgent at scale.

Structural validation tests verifying DynamicSearchFanout handles plans
with varying numbers of sub-questions (5, 10, 15). Tests verify correct
agent group counts, agents per group, state key patterns, and name
uniqueness without requiring real API calls.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from magi_system.config import ModelConfig, PipelineConfig


def _make_pipeline_config(**overrides):
    """Create a minimal PipelineConfig for testing."""
    defaults = dict(
        orchestrator=ModelConfig(
            name="gemini_pro", model="gemini-3-pro-preview", role="orchestrator"
        ),
        synthesis=ModelConfig(
            name="claude", model="mock-model", role="synthesis"
        ),
        search_models=[
            ModelConfig(name="gemini_pro", model="gemini-3-pro-preview", role="search"),
            ModelConfig(name="gpt", model="openai/gpt-5.1", role="search"),
            ModelConfig(name="grok", model="xai/grok-4-1-fast", role="search"),
        ],
        reflection_models=[
            ModelConfig(
                name="gemini_pro", model="gemini-3-pro-preview", role="reflection"
            ),
        ],
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_mock_ctx(research_plan=""):
    """Create a mock InvocationContext with session state."""
    ctx = MagicMock()
    ctx.session.state = {"research_plan": research_plan}
    return ctx


def _make_plan(n: int) -> str:
    """Generate a research plan with n sub-questions."""
    return "\n".join(f"SQ{i}: Sub-question number {i}?" for i in range(1, n + 1))


async def _collect_events(agent, ctx):
    """Run the agent's _run_async_impl and collect yielded events."""
    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)
    return events


async def _async_iter(items):
    """Create an async iterator from a list."""
    for item in items:
        yield item


class _AgentCapture:
    """Context manager that captures ParallelAgent and LlmAgent creation kwargs."""

    def __init__(self):
        self.parallel_agents = []
        self.llm_agents = []

    def __enter__(self):
        from google.adk.agents import LlmAgent, ParallelAgent

        self._orig_parallel_init = ParallelAgent.__init__
        self._orig_llm_init = LlmAgent.__init__

        capture = self

        def capture_parallel_init(self_agent, **kwargs):
            capture.parallel_agents.append(kwargs)
            capture._orig_parallel_init(self_agent, **kwargs)

        def capture_llm_init(self_agent, **kwargs):
            capture.llm_agents.append(kwargs)
            capture._orig_llm_init(self_agent, **kwargs)

        self._parallel_patch = patch.object(
            ParallelAgent, "__init__", capture_parallel_init
        )
        self._llm_patch = patch.object(LlmAgent, "__init__", capture_llm_init)
        self._run_patch = patch.object(
            ParallelAgent, "run_async", return_value=_async_iter([])
        )

        self._parallel_patch.start()
        self._llm_patch.start()
        self._run_patch.start()
        return self

    def __exit__(self, *args):
        self._run_patch.stop()
        self._llm_patch.stop()
        self._parallel_patch.stop()


@patch("magi_system.search.search_fanout.google_search")
@patch("magi_system.search.search_fanout.extract_page_content")
class TestScaleValidation5SubQuestions:
    """Validate DynamicSearchFanout with 5 sub-questions."""

    NUM_SQ = 5

    def test_correct_number_of_agent_groups(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        # Last parallel agent is the outer fanout containing sq groups
        outer_fanout = cap.parallel_agents[-1]
        assert len(outer_fanout["sub_agents"]) == self.NUM_SQ

    def test_correct_agents_per_group(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        enabled_models = [m for m in cfg.search_models if m.enabled]
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        # Each sq group (first N parallel agents) should have len(enabled_models) agents
        sq_groups = cap.parallel_agents[: self.NUM_SQ]
        for group in sq_groups:
            assert len(group["sub_agents"]) == len(enabled_models)

    def test_all_expected_state_keys(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        enabled_models = [m for m in cfg.search_models if m.enabled]
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        output_keys = {a["output_key"] for a in cap.llm_agents}
        expected_keys = {
            f"sq{i}_{m.name}" for i in range(1, self.NUM_SQ + 1) for m in enabled_models
        }
        assert output_keys == expected_keys

    def test_agent_names_are_unique(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        names = [a["name"] for a in cap.llm_agents]
        assert len(names) == len(set(names)), "Agent names must be unique"


@patch("magi_system.search.search_fanout.google_search")
@patch("magi_system.search.search_fanout.extract_page_content")
class TestScaleValidation10SubQuestions:
    """Validate DynamicSearchFanout with 10 sub-questions."""

    NUM_SQ = 10

    def test_correct_number_of_agent_groups(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        outer_fanout = cap.parallel_agents[-1]
        assert len(outer_fanout["sub_agents"]) == self.NUM_SQ

    def test_correct_agents_per_group(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        enabled_models = [m for m in cfg.search_models if m.enabled]
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        sq_groups = cap.parallel_agents[: self.NUM_SQ]
        for group in sq_groups:
            assert len(group["sub_agents"]) == len(enabled_models)

    def test_all_expected_state_keys(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        enabled_models = [m for m in cfg.search_models if m.enabled]
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        output_keys = {a["output_key"] for a in cap.llm_agents}
        expected_keys = {
            f"sq{i}_{m.name}" for i in range(1, self.NUM_SQ + 1) for m in enabled_models
        }
        assert output_keys == expected_keys

    def test_agent_names_are_unique(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        names = [a["name"] for a in cap.llm_agents]
        assert len(names) == len(set(names)), "Agent names must be unique"

    def test_total_agent_count(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        enabled_models = [m for m in cfg.search_models if m.enabled]
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        assert len(cap.llm_agents) == self.NUM_SQ * len(enabled_models)


@patch("magi_system.search.search_fanout.google_search")
@patch("magi_system.search.search_fanout.extract_page_content")
class TestScaleValidation15SubQuestions:
    """Validate DynamicSearchFanout with 15 sub-questions (max default)."""

    NUM_SQ = 15

    def test_correct_number_of_agent_groups(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        outer_fanout = cap.parallel_agents[-1]
        assert len(outer_fanout["sub_agents"]) == self.NUM_SQ

    def test_correct_agents_per_group(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        enabled_models = [m for m in cfg.search_models if m.enabled]
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        sq_groups = cap.parallel_agents[: self.NUM_SQ]
        for group in sq_groups:
            assert len(group["sub_agents"]) == len(enabled_models)

    def test_all_expected_state_keys(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        enabled_models = [m for m in cfg.search_models if m.enabled]
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        output_keys = {a["output_key"] for a in cap.llm_agents}
        expected_keys = {
            f"sq{i}_{m.name}" for i in range(1, self.NUM_SQ + 1) for m in enabled_models
        }
        assert output_keys == expected_keys

    def test_agent_names_are_unique(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        names = [a["name"] for a in cap.llm_agents]
        assert len(names) == len(set(names)), "Agent names must be unique"

    def test_total_agent_count(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        enabled_models = [m for m in cfg.search_models if m.enabled]
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        assert len(cap.llm_agents) == self.NUM_SQ * len(enabled_models)

    def test_state_key_pattern_format(self, _mock_ext, _mock_gs) -> None:
        """Verify all output keys match the sq{N}_{model} pattern."""
        import re

        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(self.NUM_SQ))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        pattern = re.compile(r"^sq\d+_\w+$")
        for a in cap.llm_agents:
            assert pattern.match(a["output_key"]), (
                f"output_key '{a['output_key']}' does not match sq{{N}}_{{model}} pattern"
            )


@patch("magi_system.search.search_fanout.google_search")
@patch("magi_system.search.search_fanout.extract_page_content")
class TestScaleValidationMaxCapping:
    """Test that max_sub_questions correctly caps at scale."""

    def test_20_questions_capped_to_15(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config(max_sub_questions=15)
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(20))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        outer_fanout = cap.parallel_agents[-1]
        assert len(outer_fanout["sub_agents"]) == 15

    def test_20_questions_capped_to_5(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config(max_sub_questions=5)
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(20))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        outer_fanout = cap.parallel_agents[-1]
        assert len(outer_fanout["sub_agents"]) == 5


@patch("magi_system.search.search_fanout.google_search")
@patch("magi_system.search.search_fanout.extract_page_content")
class TestScaleValidationMixedEnabledDisabled:
    """Test scale with a mix of enabled and disabled models."""

    def test_10_questions_with_1_of_3_models_disabled(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config(
            search_models=[
                ModelConfig(name="gemini_pro", model="gemini-3-pro-preview", role="search", enabled=True),
                ModelConfig(name="gpt", model="openai/gpt-5.1", role="search", enabled=False),
                ModelConfig(name="grok", model="xai/grok-4-1-fast", role="search", enabled=True),
            ]
        )
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan=_make_plan(10))

        with _AgentCapture() as cap:
            asyncio.get_event_loop().run_until_complete(
                _collect_events(agent, ctx)
            )

        # 10 groups, 2 enabled models each
        assert len(cap.llm_agents) == 20
        sq_groups = cap.parallel_agents[:10]
        for group in sq_groups:
            assert len(group["sub_agents"]) == 2


@pytest.mark.integration
class TestScaleValidationIntegration:
    """Integration tests requiring real API keys.

    These tests verify end-to-end behavior with actual LLM API calls.
    Run with: pytest -m integration
    """

    @pytest.mark.skip(reason="Requires real API keys and incurs cost")
    def test_5_sub_questions_real_api(self) -> None:
        """Placeholder for real API scale test with 5 sub-questions."""
        pass

    @pytest.mark.skip(reason="Requires real API keys and incurs cost")
    def test_10_sub_questions_real_api(self) -> None:
        """Placeholder for real API scale test with 10 sub-questions."""
        pass

    @pytest.mark.skip(reason="Requires real API keys and incurs cost")
    def test_15_sub_questions_real_api(self) -> None:
        """Placeholder for real API scale test with 15 sub-questions."""
        pass
