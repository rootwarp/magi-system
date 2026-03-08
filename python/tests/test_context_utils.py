"""Tests for MAGI-033: Context management utilities."""

from magi_system.tools.context_utils import truncate_text, summarize_search_results


class TestTruncateTextNoTruncation:
    """Test that short text passes through unchanged."""

    def test_short_text_returned_unchanged(self) -> None:
        text = "Hello, world!"
        result = truncate_text(text, max_chars=100)
        assert result == text

    def test_text_exactly_at_limit_returned_unchanged(self) -> None:
        text = "a" * 100
        result = truncate_text(text, max_chars=100)
        assert result == text

    def test_empty_string_returned_unchanged(self) -> None:
        result = truncate_text("", max_chars=100)
        assert result == ""


class TestTruncateTextWithTruncation:
    """Test that long text is properly truncated."""

    def test_long_text_is_truncated(self) -> None:
        text = "a" * 200
        result = truncate_text(text, max_chars=100, preserve_start=30)
        assert len(result) <= 100

    def test_truncated_text_preserves_start(self) -> None:
        start = "START" * 10  # 50 chars
        middle = "M" * 500
        end = "END" * 10  # 30 chars
        text = start + middle + end
        result = truncate_text(text, max_chars=200, preserve_start=50)
        assert result.startswith(start)

    def test_truncated_text_preserves_end(self) -> None:
        start = "S" * 50
        middle = "M" * 500
        end = "END_MARKER"
        text = start + middle + end
        result = truncate_text(text, max_chars=200, preserve_start=50)
        assert result.endswith(end)

    def test_truncated_text_includes_marker(self) -> None:
        text = "a" * 200
        result = truncate_text(text, max_chars=100, preserve_start=30)
        assert "TRUNCATED" in result

    def test_truncated_text_includes_context_limit_message(self) -> None:
        text = "a" * 200
        result = truncate_text(text, max_chars=100, preserve_start=30)
        assert "context limit" in result


class TestTruncateTextDefaults:
    """Test default parameter values."""

    def test_default_max_chars_is_50000(self) -> None:
        text = "a" * 49999
        result = truncate_text(text)
        assert result == text  # Should not truncate under default limit

    def test_default_truncates_above_50000(self) -> None:
        text = "a" * 60000
        result = truncate_text(text)
        assert len(result) <= 50000
        assert "TRUNCATED" in result


class TestSummarizeSearchResults:
    """Test summarize_search_results wrapper."""

    def test_short_results_unchanged(self) -> None:
        results = "Some search results"
        assert summarize_search_results(results) == results

    def test_long_results_truncated(self) -> None:
        results = "x" * 60000
        summarized = summarize_search_results(results)
        assert len(summarized) <= 50000
        assert "TRUNCATED" in summarized

    def test_custom_max_chars(self) -> None:
        results = "x" * 200
        summarized = summarize_search_results(results, max_chars=100)
        assert len(summarized) <= 100


class TestAggregatorTruncation:
    """Test that SearchResultsAggregator applies context truncation."""

    def test_aggregator_truncates_large_results(self) -> None:
        import asyncio
        from unittest.mock import MagicMock
        from magi_system.search.aggregator import SearchResultsAggregator

        agent = SearchResultsAggregator(name="test_aggregator")
        # Create state with results that exceed max_context_chars
        state = {
            "sq1_gemini": "a" * 30000,
            "sq2_gpt": "b" * 30000,
            "max_context_chars": 10000,
        }
        ctx = MagicMock()
        ctx.session.state = state

        async def _collect(ag, c):
            events = []
            async for event in ag._run_async_impl(c):
                events.append(event)
            return events

        asyncio.get_event_loop().run_until_complete(_collect(agent, ctx))

        result = ctx.session.state["all_search_results"]
        assert len(result) <= 10000
        assert "TRUNCATED" in result


class TestPipelineConfigMaxContextChars:
    """Test that PipelineConfig has max_context_chars field."""

    def test_pipeline_config_has_max_context_chars(self) -> None:
        from magi_system.config import PipelineConfig, ModelConfig

        config = PipelineConfig(
            orchestrator=ModelConfig(name="test", model="test", role="orchestrator"),
            synthesis=ModelConfig(name="test", model="test", role="synthesis"),
            search_models=[],
            reflection_models=[],
        )
        assert hasattr(config, "max_context_chars")
        assert config.max_context_chars == 50000

    def test_pipeline_config_max_context_chars_customizable(self) -> None:
        from magi_system.config import PipelineConfig, ModelConfig

        config = PipelineConfig(
            orchestrator=ModelConfig(name="test", model="test", role="orchestrator"),
            synthesis=ModelConfig(name="test", model="test", role="synthesis"),
            search_models=[],
            reflection_models=[],
            max_context_chars=100000,
        )
        assert config.max_context_chars == 100000
