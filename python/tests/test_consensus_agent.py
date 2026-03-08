"""Tests for MAGI-023: ConsensusAgent orchestrator."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="function")


# ---------------------------------------------------------------------------
# Mock helpers – mirrors the pattern used in test_divergence.py
# ---------------------------------------------------------------------------

def _mock_google_adk():
    """Set up mocks for google.adk modules so imports resolve without credentials."""
    class FakeBaseAgent:
        model_config: dict = {}

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

    mock_llm_agent = MagicMock()
    mock_genai_content = MagicMock()
    mock_genai_part = MagicMock()

    mock_genai_types = MagicMock()
    mock_genai_types.Content = mock_genai_content
    mock_genai_types.Part = mock_genai_part

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
    return saved, FakeBaseAgent, mock_genai_types


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
    saved, fake_base_agent, mock_genai_types = _mock_google_adk()
    # Clear cached magi_system modules so they re-import with mocks
    magi_keys = [k for k in sys.modules if k.startswith("magi_system")]
    saved_magi = {k: sys.modules.pop(k) for k in magi_keys}
    yield fake_base_agent, mock_genai_types
    # Restore everything
    for k in list(sys.modules):
        if k.startswith("magi_system"):
            del sys.modules[k]
    sys.modules.update(saved_magi)
    _restore_modules(saved)


# ---------------------------------------------------------------------------
# Test: Structure and import
# ---------------------------------------------------------------------------

class TestConsensusAgentStructure:
    """Verify ConsensusAgent is properly structured."""

    def test_importable_from_consensus_module(self):
        from magi_system.consensus import ConsensusAgent
        assert ConsensusAgent is not None

    def test_importable_from_init(self):
        mod = importlib.import_module("magi_system.consensus")
        assert hasattr(mod, "ConsensusAgent")

    def test_is_base_agent_subclass(self, mock_adk):
        fake_base_agent, _ = mock_adk
        from magi_system.consensus import ConsensusAgent
        assert issubclass(ConsensusAgent, fake_base_agent)

    def test_has_config_field(self):
        from magi_system.consensus import ConsensusAgent
        # The class should accept a config parameter
        assert "config" in ConsensusAgent.__annotations__ or hasattr(
            ConsensusAgent, "config"
        )

    def test_has_run_async_impl(self):
        from magi_system.consensus import ConsensusAgent
        assert hasattr(ConsensusAgent, "_run_async_impl")

    def test_has_arbitrary_types_allowed(self):
        from magi_system.consensus import ConsensusAgent
        assert ConsensusAgent.model_config.get("arbitrary_types_allowed") is True


# ---------------------------------------------------------------------------
# Test: Voting fast path
# ---------------------------------------------------------------------------

class TestVotingFastPath:
    """Verify fast path when divergence_map says DISCUSSION_NEEDED: no."""

    @pytest.mark.asyncio
    async def test_fast_path_sets_consensus_result(self, mock_adk):
        """When DISCUSSION_NEEDED: no, consensus_result should include voting prefix."""
        from magi_system.consensus import ConsensusAgent

        divergence_map = (
            "=== SUMMARY ===\n"
            "OVERALL_AGREEMENT_SCORE: 0.9\n"
            "DISCUSSION_NEEDED: no\n"
        )

        # Mock context
        state = {"divergence_map": divergence_map}
        mock_session = MagicMock()
        mock_session.state = state
        ctx = MagicMock()
        ctx.session = mock_session

        # Mock divergence_detector.run_async to be a no-op async gen
        async def fake_run_async(c):
            return
            yield  # make it an async generator

        with patch(
            "magi_system.consensus.divergence_detector"
        ) as mock_detector:
            mock_detector.run_async = fake_run_async

            config = MagicMock()
            agent = ConsensusAgent(name="test_consensus", config=config)

            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)

        assert "consensus_result" in state
        assert state["consensus_result"].startswith("CONSENSUS REACHED VIA VOTING")
        assert divergence_map in state["consensus_result"]

    @pytest.mark.asyncio
    async def test_fast_path_yields_voting_message(self, mock_adk):
        """Fast path should yield a content message about voting."""
        from magi_system.consensus import ConsensusAgent

        state = {"divergence_map": "DISCUSSION_NEEDED: no"}
        ctx = MagicMock()
        ctx.session.state = state

        async def fake_run_async(c):
            return
            yield

        with patch(
            "magi_system.consensus.divergence_detector"
        ) as mock_detector:
            mock_detector.run_async = fake_run_async
            agent = ConsensusAgent(name="test_consensus", config=MagicMock())

            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)

        # Should have yielded at least one content event
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Test: Discussion path
# ---------------------------------------------------------------------------

class TestDiscussionPath:
    """Verify discussion loop is triggered when DISCUSSION_NEEDED: yes."""

    @pytest.mark.asyncio
    async def test_discussion_path_creates_loop(self, mock_adk):
        """When DISCUSSION_NEEDED: yes, create_discussion_loop should be called."""
        from magi_system.consensus import ConsensusAgent

        state = {"divergence_map": "DISCUSSION_NEEDED: yes"}
        ctx = MagicMock()
        ctx.session.state = state

        async def fake_run_async(c):
            return
            yield

        mock_discussion_loop = MagicMock()
        mock_discussion_loop.run_async = fake_run_async

        with patch(
            "magi_system.consensus.divergence_detector"
        ) as mock_detector, patch(
            "magi_system.consensus.create_discussion_loop",
            return_value=mock_discussion_loop,
        ) as mock_create:
            mock_detector.run_async = fake_run_async
            config = MagicMock()
            agent = ConsensusAgent(name="test_consensus", config=config)

            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)

        mock_create.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_discussion_path_yields_discussion_message(self, mock_adk):
        """Discussion path should yield a content message about discussion."""
        _, mock_genai_types = mock_adk
        from magi_system.consensus import ConsensusAgent

        state = {"divergence_map": "DISCUSSION_NEEDED: yes"}
        ctx = MagicMock()
        ctx.session.state = state

        async def fake_run_async(c):
            return
            yield

        mock_discussion_loop = MagicMock()
        mock_discussion_loop.run_async = fake_run_async

        with patch(
            "magi_system.consensus.divergence_detector"
        ) as mock_detector, patch(
            "magi_system.consensus.create_discussion_loop",
            return_value=mock_discussion_loop,
        ):
            mock_detector.run_async = fake_run_async
            agent = ConsensusAgent(name="test_consensus", config=MagicMock())

            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)

        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Test: Case insensitivity
# ---------------------------------------------------------------------------

class TestCaseInsensitivity:
    """Verify DISCUSSION_NEEDED check is case-insensitive."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "discussion_value",
        [
            "DISCUSSION_NEEDED: YES",
            "discussion_needed: yes",
            "Discussion_Needed: Yes",
            "DISCUSSION_NEEDED: Yes",
        ],
    )
    async def test_case_insensitive_yes(self, mock_adk, discussion_value):
        from magi_system.consensus import ConsensusAgent

        state = {"divergence_map": discussion_value}
        ctx = MagicMock()
        ctx.session.state = state

        async def fake_run_async(c):
            return
            yield

        mock_discussion_loop = MagicMock()
        mock_discussion_loop.run_async = fake_run_async

        with patch(
            "magi_system.consensus.divergence_detector"
        ) as mock_detector, patch(
            "magi_system.consensus.create_discussion_loop",
            return_value=mock_discussion_loop,
        ) as mock_create:
            mock_detector.run_async = fake_run_async
            agent = ConsensusAgent(name="test_consensus", config=MagicMock())

            async for _ in agent._run_async_impl(ctx):
                pass

        mock_create.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "discussion_value",
        [
            "DISCUSSION_NEEDED: NO",
            "discussion_needed: no",
            "Discussion_Needed: No",
        ],
    )
    async def test_case_insensitive_no(self, mock_adk, discussion_value):
        from magi_system.consensus import ConsensusAgent

        state = {"divergence_map": discussion_value}
        ctx = MagicMock()
        ctx.session.state = state

        async def fake_run_async(c):
            return
            yield

        with patch(
            "magi_system.consensus.divergence_detector"
        ) as mock_detector:
            mock_detector.run_async = fake_run_async
            agent = ConsensusAgent(name="test_consensus", config=MagicMock())

            async for _ in agent._run_async_impl(ctx):
                pass

        assert "consensus_result" in state
        assert state["consensus_result"].startswith("CONSENSUS REACHED VIA VOTING")
