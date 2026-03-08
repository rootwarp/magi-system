"""Tests for consensus module integration and cross-module references (MAGI-029).

These tests complement test_consensus_agent.py, test_divergence.py, and
test_discussion.py by focusing on:
- Module __all__ exports
- Cross-module wiring (ConsensusAgent references divergence_detector, create_discussion_loop)
- Edge cases in state handling (empty/missing divergence_map)
"""

import sys
from unittest.mock import MagicMock, patch

import pytest



# ---------------------------------------------------------------------------
# Mock helpers – same pattern as test_consensus_agent.py
# ---------------------------------------------------------------------------

def _mock_google_adk():
    class FakeBaseAgent:
        model_config: dict = {}

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

    mock_llm_agent = MagicMock()
    mock_genai_types = MagicMock()

    adk_modules = {
        "google": MagicMock(),
        "google.adk": MagicMock(),
        "google.adk.agents": MagicMock(
            LlmAgent=mock_llm_agent,
            BaseAgent=FakeBaseAgent,
            LoopAgent=MagicMock(),
            SequentialAgent=MagicMock(),
            ParallelAgent=MagicMock(),
        ),
        "google.adk.agents.invocation_context": MagicMock(),
        "google.adk.events": MagicMock(),
        "google.adk.events.event": MagicMock(),
        "google.adk.models": MagicMock(),
        "google.adk.models.lite_llm": MagicMock(),
        "google.adk.tools": MagicMock(),
        "google.adk.tools.agent_tool": MagicMock(),
        "google.adk.tools.exit_loop_tool": MagicMock(),
        "google.adk.tools.google_search": MagicMock(),
        "google.genai": MagicMock(),
        "google.genai.types": mock_genai_types,
    }
    saved = {}
    for name, mock in adk_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock
    return saved, FakeBaseAgent


def _restore_modules(saved):
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture(autouse=True)
def mock_adk():
    saved, fake_base_agent = _mock_google_adk()
    magi_keys = [k for k in sys.modules if k.startswith("magi_system")]
    saved_magi = {k: sys.modules.pop(k) for k in magi_keys}
    yield fake_base_agent
    for k in list(sys.modules):
        if k.startswith("magi_system"):
            del sys.modules[k]
    sys.modules.update(saved_magi)
    _restore_modules(saved)


async def _fake_async_gen(c):
    """Helper async generator that yields nothing."""
    return
    yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Tests: Module __all__ exports
# ---------------------------------------------------------------------------

class TestConsensusModuleExports:
    """Verify __all__ contains exactly the expected public names."""

    def test_all_contains_consensus_agent(self):
        from magi_system.consensus import __all__

        assert "ConsensusAgent" in __all__

    def test_all_contains_divergence_detector(self):
        from magi_system.consensus import __all__

        assert "divergence_detector" in __all__

    def test_all_contains_create_discussion_loop(self):
        from magi_system.consensus import __all__

        assert "create_discussion_loop" in __all__

    def test_all_has_three_exports(self):
        from magi_system.consensus import __all__

        assert len(__all__) == 3


# ---------------------------------------------------------------------------
# Tests: Cross-module wiring
# ---------------------------------------------------------------------------

class TestCrossModuleReferences:
    """Verify ConsensusAgent correctly wires to divergence_detector and discussion_loop."""

    def test_consensus_module_exposes_divergence_detector(self):
        """divergence_detector should be importable from consensus package."""
        from magi_system.consensus import divergence_detector

        assert divergence_detector is not None

    def test_consensus_module_exposes_create_discussion_loop(self):
        from magi_system.consensus import create_discussion_loop

        assert callable(create_discussion_loop)

    def test_consensus_agent_config_annotation_is_pipeline_config(self):
        from magi_system.consensus import ConsensusAgent

        annotations = ConsensusAgent.__annotations__
        assert "config" in annotations

    def test_divergence_detector_output_key_matches_consensus_agent_state_read(self):
        """ConsensusAgent reads 'divergence_map' from state, which is divergence_detector's output_key."""
        # divergence_detector is a mock here, but we can verify the source code contract
        # by checking that ConsensusAgent._run_async_impl reads "divergence_map"
        import inspect
        from magi_system.consensus import ConsensusAgent

        source = inspect.getsource(ConsensusAgent._run_async_impl)
        assert "divergence_map" in source


# ---------------------------------------------------------------------------
# Tests: Edge cases in state handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio(loop_scope="function")
class TestStateEdgeCases:
    """Edge cases for divergence_map state variable."""

    async def test_missing_divergence_map_defaults_to_empty(self):
        """When divergence_map is not in state, should default to '' and take fast path."""
        from magi_system.consensus import ConsensusAgent

        state = {}  # no divergence_map key
        ctx = MagicMock()
        ctx.session.state = state

        with patch(
            "magi_system.consensus.divergence_detector"
        ) as mock_detector:
            mock_detector.run_async = _fake_async_gen
            agent = ConsensusAgent(name="test_consensus", config=MagicMock())

            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)

        # Should take fast path since empty string doesn't contain "discussion_needed: yes"
        assert "consensus_result" in state
        assert state["consensus_result"].startswith("CONSENSUS REACHED VIA VOTING")

    async def test_empty_divergence_map_takes_fast_path(self):
        """An empty divergence_map string should take the voting fast path."""
        from magi_system.consensus import ConsensusAgent

        state = {"divergence_map": ""}
        ctx = MagicMock()
        ctx.session.state = state

        with patch(
            "magi_system.consensus.divergence_detector"
        ) as mock_detector:
            mock_detector.run_async = _fake_async_gen
            agent = ConsensusAgent(name="test_consensus", config=MagicMock())

            async for _ in agent._run_async_impl(ctx):
                pass

        assert state["consensus_result"].startswith("CONSENSUS REACHED VIA VOTING")

    async def test_divergence_map_with_only_whitespace_takes_fast_path(self):
        """Whitespace-only divergence_map should take fast path."""
        from magi_system.consensus import ConsensusAgent

        state = {"divergence_map": "   \n\n  "}
        ctx = MagicMock()
        ctx.session.state = state

        with patch(
            "magi_system.consensus.divergence_detector"
        ) as mock_detector:
            mock_detector.run_async = _fake_async_gen
            agent = ConsensusAgent(name="test_consensus", config=MagicMock())

            async for _ in agent._run_async_impl(ctx):
                pass

        assert state["consensus_result"].startswith("CONSENSUS REACHED VIA VOTING")

    async def test_consensus_result_includes_divergence_map_content(self):
        """Fast path consensus_result should embed the full divergence_map."""
        from magi_system.consensus import ConsensusAgent

        divergence_text = "OVERALL_AGREEMENT_SCORE: 0.95\nDISCUSSION_NEEDED: no"
        state = {"divergence_map": divergence_text}
        ctx = MagicMock()
        ctx.session.state = state

        with patch(
            "magi_system.consensus.divergence_detector"
        ) as mock_detector:
            mock_detector.run_async = _fake_async_gen
            agent = ConsensusAgent(name="test_consensus", config=MagicMock())

            async for _ in agent._run_async_impl(ctx):
                pass

        assert divergence_text in state["consensus_result"]
