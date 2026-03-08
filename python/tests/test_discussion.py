"""Tests for MAGI-022: Discussion loop with AgentTool."""

import sys
from unittest.mock import MagicMock

import pytest


# Track all agent classes for inspection
_mock_classes = {}


def _mock_google_adk():
    """Set up mocks for google.adk modules so imports resolve without credentials."""

    class FakeLlmAgent:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeLoopAgent:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeSequentialAgent:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeAgentTool:
        def __init__(self, agent=None, **kwargs):
            self.agent = agent
            self._kwargs = kwargs

    class FakeLiteLlm:
        def __init__(self, model=None, **kwargs):
            self.model = model

    fake_exit_loop = MagicMock()
    fake_exit_loop.__name__ = "exit_loop"

    _mock_classes["LlmAgent"] = FakeLlmAgent
    _mock_classes["LoopAgent"] = FakeLoopAgent
    _mock_classes["SequentialAgent"] = FakeSequentialAgent
    _mock_classes["AgentTool"] = FakeAgentTool
    _mock_classes["LiteLlm"] = FakeLiteLlm
    _mock_classes["exit_loop"] = fake_exit_loop

    mock_agents = MagicMock()
    mock_agents.LlmAgent = FakeLlmAgent
    mock_agents.LoopAgent = FakeLoopAgent
    mock_agents.SequentialAgent = FakeSequentialAgent

    mock_agent_tool_mod = MagicMock()
    mock_agent_tool_mod.AgentTool = FakeAgentTool

    mock_exit_loop_mod = MagicMock()
    mock_exit_loop_mod.exit_loop = fake_exit_loop

    mock_lite_llm_mod = MagicMock()
    mock_lite_llm_mod.LiteLlm = FakeLiteLlm

    adk_modules = {
        "google": MagicMock(),
        "google.adk": MagicMock(),
        "google.adk.agents": mock_agents,
        "google.adk.models": MagicMock(),
        "google.adk.models.lite_llm": mock_lite_llm_mod,
        "google.adk.tools": MagicMock(),
        "google.adk.tools.agent_tool": mock_agent_tool_mod,
        "google.adk.tools.exit_loop_tool": mock_exit_loop_mod,
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
    magi_keys = [k for k in sys.modules if k.startswith("magi_system")]
    saved_magi = {k: sys.modules.pop(k) for k in magi_keys}
    yield
    for k in list(sys.modules):
        if k.startswith("magi_system"):
            del sys.modules[k]
    sys.modules.update(saved_magi)
    _restore_modules(saved)


def _make_config(num_models=3):
    """Create a PipelineConfig for testing."""
    from magi_system.config import ModelConfig, PipelineConfig

    search_models = []
    names = [
        ("gemini_pro", "gemini-3-pro-preview"),
        ("gpt", "openai/gpt-5.1"),
        ("grok", "xai/grok-4-1-fast"),
    ]
    for i in range(num_models):
        name, model = names[i % len(names)]
        search_models.append(
            ModelConfig(name=name, model=model, role="search", enabled=True)
        )

    return PipelineConfig(
        orchestrator=ModelConfig(
            name="gemini_pro", model="gemini-3-pro-preview", role="orchestrator"
        ),
        synthesis=ModelConfig(name="claude", model="claude-opus", role="synthesis"),
        search_models=search_models,
        reflection_models=[],
        max_discussion_rounds=5,
    )


class TestCreateDiscussionLoopReturnsLoopAgent:
    """Test that create_discussion_loop returns a LoopAgent."""

    def test_returns_loop_agent(self):
        from magi_system.consensus.discussion import create_discussion_loop

        config = _make_config()
        result = create_discussion_loop(config)
        assert isinstance(result, _mock_classes["LoopAgent"])

    def test_loop_agent_max_iterations_from_config(self):
        from magi_system.consensus.discussion import create_discussion_loop

        config = _make_config()
        config.max_discussion_rounds = 7
        result = create_discussion_loop(config)
        assert result.max_iterations == 7


class TestLoopAgentStructure:
    """Test the internal structure of the discussion loop."""

    def test_loop_contains_sequential_sub_agent(self):
        from magi_system.consensus.discussion import create_discussion_loop

        config = _make_config()
        result = create_discussion_loop(config)
        assert len(result.sub_agents) == 1
        assert isinstance(result.sub_agents[0], _mock_classes["SequentialAgent"])

    def test_sequential_agent_has_two_sub_agents(self):
        from magi_system.consensus.discussion import create_discussion_loop

        config = _make_config()
        result = create_discussion_loop(config)
        seq = result.sub_agents[0]
        assert len(seq.sub_agents) == 2


class TestDiscussionRouter:
    """Test the discussion_router LlmAgent."""

    def _get_router(self):
        from magi_system.consensus.discussion import create_discussion_loop

        config = _make_config()
        result = create_discussion_loop(config)
        return result.sub_agents[0].sub_agents[0]

    def test_router_is_llm_agent(self):
        router = self._get_router()
        assert isinstance(router, _mock_classes["LlmAgent"])

    def test_router_output_key(self):
        router = self._get_router()
        assert router.output_key == "discussion_round_result"

    def test_router_has_agent_tools(self):
        router = self._get_router()
        assert len(router.tools) > 0
        for tool in router.tools:
            assert isinstance(tool, _mock_classes["AgentTool"])

    def test_router_instruction_contains_divergence_map(self):
        router = self._get_router()
        assert "{divergence_map}" in router.instruction


class TestResolutionEvaluator:
    """Test the resolution_evaluator LlmAgent."""

    def _get_evaluator(self):
        from magi_system.consensus.discussion import create_discussion_loop

        config = _make_config()
        result = create_discussion_loop(config)
        return result.sub_agents[0].sub_agents[1]

    def test_evaluator_is_llm_agent(self):
        evaluator = self._get_evaluator()
        assert isinstance(evaluator, _mock_classes["LlmAgent"])

    def test_evaluator_output_key(self):
        evaluator = self._get_evaluator()
        assert evaluator.output_key == "consensus_result"

    def test_evaluator_has_exit_loop_tool(self):
        evaluator = self._get_evaluator()
        assert _mock_classes["exit_loop"] in evaluator.tools

    def test_evaluator_instruction_contains_divergence_map(self):
        evaluator = self._get_evaluator()
        assert "{divergence_map}" in evaluator.instruction

    def test_evaluator_instruction_contains_discussion_round_result(self):
        evaluator = self._get_evaluator()
        assert "{discussion_round_result}" in evaluator.instruction


class TestFactoryCreatesFreshAgents:
    """Test that the factory creates new instances each call."""

    def test_different_instances_on_each_call(self):
        from magi_system.consensus.discussion import create_discussion_loop

        config = _make_config()
        result1 = create_discussion_loop(config)
        result2 = create_discussion_loop(config)
        assert result1 is not result2

    def test_inner_agents_are_different_instances(self):
        from magi_system.consensus.discussion import create_discussion_loop

        config = _make_config()
        result1 = create_discussion_loop(config)
        result2 = create_discussion_loop(config)
        router1 = result1.sub_agents[0].sub_agents[0]
        router2 = result2.sub_agents[0].sub_agents[0]
        assert router1 is not router2


class TestDisabledModelsExcluded:
    """Test that disabled models are excluded from discussion agents."""

    def test_disabled_model_not_in_tools(self):
        from magi_system.config import ModelConfig, PipelineConfig
        from magi_system.consensus.discussion import create_discussion_loop

        config = PipelineConfig(
            orchestrator=ModelConfig(
                name="gemini_pro", model="gemini-3-pro-preview", role="orchestrator"
            ),
            synthesis=ModelConfig(
                name="claude", model="claude-opus", role="synthesis"
            ),
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
            ],
            reflection_models=[],
        )
        result = create_discussion_loop(config)
        router = result.sub_agents[0].sub_agents[0]
        assert len(router.tools) == 1


class TestImportability:
    """Test that create_discussion_loop is importable from the expected paths."""

    def test_import_from_discussion_module(self):
        from magi_system.consensus.discussion import create_discussion_loop

        assert create_discussion_loop is not None

    def test_import_from_consensus_package(self):
        from magi_system.consensus import create_discussion_loop

        assert create_discussion_loop is not None
