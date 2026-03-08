"""Tests for MAGI-019: SearchResultsAggregator BaseAgent."""

import asyncio
from unittest.mock import MagicMock

from google.adk.agents import BaseAgent


def _make_mock_ctx(state=None):
    """Create a mock InvocationContext with session state."""
    ctx = MagicMock()
    ctx.session.state = state if state is not None else {}
    return ctx


async def _collect_events(agent, ctx):
    """Run the agent's _run_async_impl and collect yielded events."""
    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)
    return events


class TestSearchResultsAggregatorStructure:
    """Structural tests for SearchResultsAggregator."""

    def test_is_base_agent_subclass(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        assert issubclass(SearchResultsAggregator, BaseAgent)

    def test_has_run_async_impl_method(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        assert hasattr(SearchResultsAggregator, "_run_async_impl")

    def test_has_arbitrary_types_allowed(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        model_config = SearchResultsAggregator.model_config
        assert model_config.get("arbitrary_types_allowed") is True

    def test_can_instantiate(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        assert agent.name == "test_aggregator"


class TestSearchResultsAggregatorExport:
    """Test that SearchResultsAggregator is exported from the search package."""

    def test_importable_from_search_package(self) -> None:
        from magi_system.search import SearchResultsAggregator

        assert SearchResultsAggregator is not None


class TestSearchResultsAggregatorEmptyState:
    """Test behavior when state has no search results."""

    def test_empty_state_writes_no_results_message(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        ctx = _make_mock_ctx(state={})

        asyncio.get_event_loop().run_until_complete(_collect_events(agent, ctx))

        assert ctx.session.state["all_search_results"] == "No search results available."

    def test_empty_state_yields_content(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        ctx = _make_mock_ctx(state={})

        events = asyncio.get_event_loop().run_until_complete(
            _collect_events(agent, ctx)
        )

        assert len(events) == 1
        assert events[0].role == "model"
        assert "0" in events[0].parts[0].text


class TestSearchResultsAggregatorAggregation:
    """Test aggregation logic with mock state containing sq keys."""

    def test_aggregates_single_result(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        state = {"sq1_gemini": "Gemini found evidence about quantum computing."}
        ctx = _make_mock_ctx(state=state)

        asyncio.get_event_loop().run_until_complete(_collect_events(agent, ctx))

        result = ctx.session.state["all_search_results"]
        assert "Sub-Question 1" in result
        assert "gemini" in result
        assert "Gemini found evidence about quantum computing." in result

    def test_aggregates_multiple_results(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        state = {
            "sq1_gemini": "Gemini result for Q1",
            "sq1_gpt": "GPT result for Q1",
            "sq2_gemini": "Gemini result for Q2",
        }
        ctx = _make_mock_ctx(state=state)

        asyncio.get_event_loop().run_until_complete(_collect_events(agent, ctx))

        result = ctx.session.state["all_search_results"]
        assert "Sub-Question 1" in result
        assert "Sub-Question 2" in result
        assert "gemini" in result
        assert "gpt" in result
        assert "Gemini result for Q1" in result
        assert "GPT result for Q1" in result
        assert "Gemini result for Q2" in result

    def test_results_have_clear_headers(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        state = {
            "sq1_gemini": "Result A",
            "sq2_gpt": "Result B",
        }
        ctx = _make_mock_ctx(state=state)

        asyncio.get_event_loop().run_until_complete(_collect_events(agent, ctx))

        result = ctx.session.state["all_search_results"]
        # Each result section should have a markdown header
        assert "## Sub-Question 1" in result
        assert "## Sub-Question 2" in result

    def test_results_separated_by_dividers(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        state = {
            "sq1_gemini": "Result A",
            "sq2_gpt": "Result B",
        }
        ctx = _make_mock_ctx(state=state)

        asyncio.get_event_loop().run_until_complete(_collect_events(agent, ctx))

        result = ctx.session.state["all_search_results"]
        assert "---" in result

    def test_event_reports_count(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        state = {
            "sq1_gemini": "Result A",
            "sq1_gpt": "Result B",
            "sq2_gemini": "Result C",
        }
        ctx = _make_mock_ctx(state=state)

        events = asyncio.get_event_loop().run_until_complete(
            _collect_events(agent, ctx)
        )

        assert len(events) == 1
        assert "3" in events[0].parts[0].text


class TestSearchResultsAggregatorMissingKeys:
    """Test handling of missing or empty results."""

    def test_skips_empty_values(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        state = {
            "sq1_gemini": "Good result",
            "sq1_gpt": "",
            "sq2_gemini": None,
        }
        ctx = _make_mock_ctx(state=state)

        asyncio.get_event_loop().run_until_complete(_collect_events(agent, ctx))

        result = ctx.session.state["all_search_results"]
        assert "Good result" in result
        # Empty and None values should be skipped
        assert result.count("## Sub-Question") == 1

    def test_ignores_non_sq_keys(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        state = {
            "research_plan": "Some plan",
            "sq1_gemini": "Actual result",
            "other_key": "Should be ignored",
        }
        ctx = _make_mock_ctx(state=state)

        asyncio.get_event_loop().run_until_complete(_collect_events(agent, ctx))

        result = ctx.session.state["all_search_results"]
        assert "Actual result" in result
        assert "Some plan" not in result
        assert "Should be ignored" not in result

    def test_handles_model_names_with_underscores(self) -> None:
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        state = {
            "sq1_gemini_pro": "Result from gemini pro",
        }
        ctx = _make_mock_ctx(state=state)

        asyncio.get_event_loop().run_until_complete(_collect_events(agent, ctx))

        result = ctx.session.state["all_search_results"]
        assert "gemini_pro" in result
        assert "Result from gemini pro" in result
