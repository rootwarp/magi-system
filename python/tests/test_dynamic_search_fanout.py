"""Tests for MAGI-013: DynamicSearchFanout BaseAgent."""

import asyncio
from unittest.mock import MagicMock, patch


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


class TestDynamicSearchFanoutStructure:
    """Structural tests for DynamicSearchFanout."""

    def test_is_base_agent_subclass(self) -> None:
        from google.adk.agents import BaseAgent

        from magi_system.search.search_fanout import DynamicSearchFanout

        assert issubclass(DynamicSearchFanout, BaseAgent)

    def test_has_config_field(self) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        assert "config" in DynamicSearchFanout.model_fields

    def test_has_arbitrary_types_allowed(self) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        model_config = DynamicSearchFanout.model_config
        assert model_config.get("arbitrary_types_allowed") is True

    def test_can_instantiate_with_pipeline_config(self) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        assert agent.config is cfg
        assert agent.name == "test_fanout"


class TestDynamicSearchFanoutExport:
    """Test that DynamicSearchFanout is exported from the search package."""

    def test_importable_from_search_package(self) -> None:
        from magi_system.search import DynamicSearchFanout

        assert DynamicSearchFanout is not None


class TestDynamicSearchFanoutEmptyPlan:
    """Test behavior when research_plan is empty or has no sub-questions."""

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_empty_plan_yields_nothing(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = _make_mock_ctx(research_plan="")

        events = asyncio.get_event_loop().run_until_complete(
            _collect_events(agent, ctx)
        )
        assert events == []

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_no_research_plan_key_yields_nothing(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)
        ctx = MagicMock()
        ctx.session.state = {}

        events = asyncio.get_event_loop().run_until_complete(
            _collect_events(agent, ctx)
        )
        assert events == []


class TestDynamicSearchFanoutAgentCreation:
    """Test agent creation logic with mocked sub-questions."""

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_creates_parallel_agents_for_sub_questions(
        self, _mock_ext, _mock_gs
    ) -> None:
        from google.adk.agents import ParallelAgent

        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config()
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)

        plan = "SQ1: What is quantum computing?\nSQ2: How does AI work?"
        ctx = _make_mock_ctx(research_plan=plan)

        created_fanouts = []
        original_init = ParallelAgent.__init__

        def capture_init(self, **kwargs):
            created_fanouts.append(kwargs)
            original_init(self, **kwargs)

        with patch.object(ParallelAgent, "__init__", capture_init):
            with patch.object(
                ParallelAgent, "run_async", return_value=_async_iter([])
            ):
                asyncio.get_event_loop().run_until_complete(
                    _collect_events(agent, ctx)
                )

        # Outer fanout should have 2 sq groups (one per sub-question)
        outer_fanout = created_fanouts[-1]
        assert len(outer_fanout["sub_agents"]) == 2

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_caps_sub_questions_to_max(self, _mock_ext, _mock_gs) -> None:
        from google.adk.agents import ParallelAgent

        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config(max_sub_questions=1)
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)

        plan = "SQ1: Question one\nSQ2: Question two\nSQ3: Question three"
        ctx = _make_mock_ctx(research_plan=plan)

        created_fanouts = []
        original_init = ParallelAgent.__init__

        def capture_init(self, **kwargs):
            created_fanouts.append(kwargs)
            original_init(self, **kwargs)

        with patch.object(ParallelAgent, "__init__", capture_init):
            with patch.object(
                ParallelAgent, "run_async", return_value=_async_iter([])
            ):
                asyncio.get_event_loop().run_until_complete(
                    _collect_events(agent, ctx)
                )

        # The outer fanout should have only 1 sub-agent group (capped to 1)
        outer_fanout = created_fanouts[-1]
        assert len(outer_fanout["sub_agents"]) == 1

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_skips_disabled_models(self, _mock_ext, _mock_gs) -> None:
        from google.adk.agents import ParallelAgent

        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config(
            search_models=[
                ModelConfig(
                    name="gemini_pro",
                    model="gemini-3-pro-preview",
                    role="search",
                    enabled=True,
                ),
                ModelConfig(
                    name="gpt",
                    model="openai/gpt-5.1",
                    role="search",
                    enabled=False,
                ),
            ]
        )
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)

        plan = "SQ1: What is quantum computing?"
        ctx = _make_mock_ctx(research_plan=plan)

        created_groups = []
        original_init = ParallelAgent.__init__

        def capture_init(self, **kwargs):
            created_groups.append(kwargs)
            original_init(self, **kwargs)

        with patch.object(ParallelAgent, "__init__", capture_init):
            with patch.object(
                ParallelAgent, "run_async", return_value=_async_iter([])
            ):
                asyncio.get_event_loop().run_until_complete(
                    _collect_events(agent, ctx)
                )

        # The SQ group should have only 1 agent (gpt is disabled)
        sq_group = created_groups[0]
        assert len(sq_group["sub_agents"]) == 1

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_agent_names_contain_run_id(self, _mock_ext, _mock_gs) -> None:
        from google.adk.agents import LlmAgent, ParallelAgent

        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config(
            search_models=[
                ModelConfig(
                    name="gemini_pro",
                    model="gemini-3-pro-preview",
                    role="search",
                ),
            ]
        )
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)

        plan = "SQ1: What is quantum computing?"
        ctx = _make_mock_ctx(research_plan=plan)

        created_llm_agents = []
        original_llm_init = LlmAgent.__init__

        def capture_llm_init(self, **kwargs):
            created_llm_agents.append(kwargs)
            original_llm_init(self, **kwargs)

        with patch.object(LlmAgent, "__init__", capture_llm_init):
            with patch.object(
                ParallelAgent, "run_async", return_value=_async_iter([])
            ):
                asyncio.get_event_loop().run_until_complete(
                    _collect_events(agent, ctx)
                )

        assert len(created_llm_agents) == 1
        name = created_llm_agents[0]["name"]
        # Name format: search_{model_name}_{sq_id}_{run_id}
        assert name.startswith("search_gemini_pro_sq1_")
        # run_id is 8 hex chars from token_hex(4)
        run_id_part = name.split("_")[-1]
        assert len(run_id_part) == 8

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_output_keys_follow_pattern(self, _mock_ext, _mock_gs) -> None:
        from google.adk.agents import LlmAgent, ParallelAgent

        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config(
            search_models=[
                ModelConfig(
                    name="gemini_pro",
                    model="gemini-3-pro-preview",
                    role="search",
                ),
                ModelConfig(
                    name="gpt",
                    model="openai/gpt-5.1",
                    role="search",
                ),
            ]
        )
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)

        plan = "SQ1: What is quantum computing?"
        ctx = _make_mock_ctx(research_plan=plan)

        created_llm_agents = []
        original_llm_init = LlmAgent.__init__

        def capture_llm_init(self, **kwargs):
            created_llm_agents.append(kwargs)
            original_llm_init(self, **kwargs)

        with patch.object(LlmAgent, "__init__", capture_llm_init):
            with patch.object(
                ParallelAgent, "run_async", return_value=_async_iter([])
            ):
                asyncio.get_event_loop().run_until_complete(
                    _collect_events(agent, ctx)
                )

        output_keys = {a["output_key"] for a in created_llm_agents}
        assert output_keys == {"sq1_gemini_pro", "sq1_gpt"}


class TestDynamicSearchFanoutAllDisabled:
    """Test behavior when all search models are disabled."""

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_all_disabled_yields_nothing(self, _mock_ext, _mock_gs) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        cfg = _make_pipeline_config(
            search_models=[
                ModelConfig(
                    name="gpt",
                    model="openai/gpt-5.1",
                    role="search",
                    enabled=False,
                ),
            ]
        )
        agent = DynamicSearchFanout(name="test_fanout", config=cfg)

        plan = "SQ1: What is quantum computing?"
        ctx = _make_mock_ctx(research_plan=plan)

        events = asyncio.get_event_loop().run_until_complete(
            _collect_events(agent, ctx)
        )
        assert events == []


# --- Helpers ---


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
