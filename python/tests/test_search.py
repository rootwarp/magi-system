"""Tests for MAGI-017: Comprehensive search module unit tests.

This module consolidates and extends tests for the search module.
Existing detailed tests live in:
  - test_sub_question_parser.py (_parse_sub_questions edge cases)
  - test_search_agent_factory.py (_get_tools_for_model, _get_model_for_search)
  - test_dynamic_search_fanout.py (DynamicSearchFanout structural + creation)

This file adds coverage for gaps not addressed in those files.
"""

from unittest.mock import MagicMock, patch

import pytest

from magi_system.config import ModelConfig, PipelineConfig
from magi_system.search.search_fanout import _parse_sub_questions


# ---------------------------------------------------------------------------
# _parse_sub_questions — additional edge cases
# ---------------------------------------------------------------------------


class TestParseSubQuestionsAdditional:
    """Additional edge cases not covered in test_sub_question_parser.py."""

    def test_none_like_input_empty_string(self) -> None:
        """Passing empty string returns empty list."""
        assert _parse_sub_questions("") == []

    def test_only_metadata_lines(self) -> None:
        """Lines that look like metadata but not SQ lines are ignored."""
        plan = "PRIORITY: high\nDEPENDS_ON: SQ1\nCOMPLEXITY: medium\n"
        assert _parse_sub_questions(plan) == []

    def test_sq_prefix_in_middle_of_line_ignored(self) -> None:
        """SQ pattern must be at start of line (possibly with whitespace)."""
        plan = "Some text SQ1: not a question\nSQ2: Real question?\n"
        result = _parse_sub_questions(plan)
        assert result == ["Real question?"]

    def test_consecutive_numbering_not_required(self) -> None:
        """Parser handles non-sequential numbering."""
        plan = "SQ1: First?\nSQ5: Fifth?\nSQ100: Hundredth?\n"
        result = _parse_sub_questions(plan)
        assert len(result) == 3
        assert result[0] == "First?"
        assert result[2] == "Hundredth?"

    def test_tabs_as_whitespace(self) -> None:
        """Tab characters in leading whitespace are handled."""
        plan = "\tSQ1: Tabbed question?\n"
        result = _parse_sub_questions(plan)
        assert result == ["Tabbed question?"]

    def test_windows_line_endings(self) -> None:
        """CRLF line endings are handled."""
        plan = "SQ1: First?\r\nSQ2: Second?\r\n"
        result = _parse_sub_questions(plan)
        assert result == ["First?", "Second?"]


# ---------------------------------------------------------------------------
# _get_tools_for_model — additional coverage
# ---------------------------------------------------------------------------


class TestGetToolsForModelAdditional:
    """Additional _get_tools_for_model tests not in test_search_agent_factory.py."""

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_gemini_flash_variant_gets_google_search(
        self, mock_extract: MagicMock, mock_gs: MagicMock
    ) -> None:
        from magi_system.search.search_fanout import _get_tools_for_model

        cfg = ModelConfig(name="gemini_flash", model="gemini-2.0-flash", role="search")
        tools = _get_tools_for_model(cfg)
        assert mock_gs in tools
        assert len(tools) == 2

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_unknown_model_gets_only_extract(
        self, mock_extract: MagicMock, mock_gs: MagicMock
    ) -> None:
        from magi_system.search.search_fanout import _get_tools_for_model

        cfg = ModelConfig(name="custom_model", model="custom/model-v1", role="search")
        tools = _get_tools_for_model(cfg)
        assert mock_extract in tools
        assert mock_gs not in tools
        assert len(tools) == 1

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_tools_list_is_new_instance_each_call(
        self, mock_extract: MagicMock, mock_gs: MagicMock
    ) -> None:
        from magi_system.search.search_fanout import _get_tools_for_model

        cfg = ModelConfig(name="gpt", model="openai/gpt-5.1", role="search")
        tools1 = _get_tools_for_model(cfg)
        tools2 = _get_tools_for_model(cfg)
        assert tools1 is not tools2


# ---------------------------------------------------------------------------
# _get_model_for_search — additional coverage
# ---------------------------------------------------------------------------


class TestGetModelForSearchAdditional:
    """Additional _get_model_for_search tests."""

    def test_grok_with_string_model_returns_litellm(self) -> None:
        """Grok with a plain string model ID gets wrapped in LiteLlm."""
        from google.adk.models.lite_llm import LiteLlm

        from magi_system.search.search_fanout import _get_model_for_search

        cfg = ModelConfig(name="grok", model="xai/grok-4-1-fast", role="search")
        result = _get_model_for_search(cfg)
        assert isinstance(result, LiteLlm)

    def test_grok_extra_body_has_search_parameters(self) -> None:
        from google.adk.models.lite_llm import LiteLlm

        from magi_system.search.search_fanout import _get_model_for_search

        cfg = ModelConfig(
            name="grok",
            model=LiteLlm(model="xai/grok-4-1-fast"),
            role="search",
        )
        result = _get_model_for_search(cfg)
        assert result._additional_args["extra_body"]["search_parameters"] == {
            "mode": "auto"
        }

    def test_non_grok_string_model_passes_through(self) -> None:
        from magi_system.search.search_fanout import _get_model_for_search

        cfg = ModelConfig(name="gemini_pro", model="gemini-3-pro-preview", role="search")
        result = _get_model_for_search(cfg)
        assert result == "gemini-3-pro-preview"
        assert isinstance(result, str)

    def test_non_grok_litellm_model_passes_through(self) -> None:
        from google.adk.models.lite_llm import LiteLlm

        from magi_system.search.search_fanout import _get_model_for_search

        original = LiteLlm(model="openai/gpt-5.1")
        cfg = ModelConfig(name="gpt", model=original, role="search")
        result = _get_model_for_search(cfg)
        assert result is original


# ---------------------------------------------------------------------------
# DynamicSearchFanout — structural tests
# ---------------------------------------------------------------------------


class TestDynamicSearchFanoutStructural:
    """Structural validation of DynamicSearchFanout class."""

    def test_is_base_agent_subclass(self) -> None:
        from google.adk.agents import BaseAgent

        from magi_system.search.search_fanout import DynamicSearchFanout

        assert issubclass(DynamicSearchFanout, BaseAgent)

    def test_has_config_field(self) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        assert "config" in DynamicSearchFanout.model_fields

    def test_config_field_type_is_pipeline_config(self) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        field = DynamicSearchFanout.model_fields["config"]
        assert field.annotation is PipelineConfig

    def test_importable_from_package(self) -> None:
        from magi_system.search import DynamicSearchFanout

        assert DynamicSearchFanout is not None

    def test_has_run_async_impl(self) -> None:
        from magi_system.search.search_fanout import DynamicSearchFanout

        assert hasattr(DynamicSearchFanout, "_run_async_impl")

    def test_instruction_template_has_required_placeholders(self) -> None:
        from magi_system.search.search_fanout import SEARCH_INSTRUCTION_TEMPLATE

        assert "{research_brief}" in SEARCH_INSTRUCTION_TEMPLATE
        assert "{sub_question}" in SEARCH_INSTRUCTION_TEMPLATE
