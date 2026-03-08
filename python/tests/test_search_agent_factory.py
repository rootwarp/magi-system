"""Tests for MAGI-011: Search agent factory in search_fanout module."""

from unittest.mock import MagicMock, patch

from magi_system.config import ModelConfig


class TestGetToolsForModel:
    """Tests for _get_tools_for_model function."""

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_gemini_model_gets_google_search_and_extract(
        self, mock_extract: MagicMock, mock_google_search: MagicMock
    ) -> None:
        from magi_system.search.search_fanout import _get_tools_for_model

        cfg = ModelConfig(
            name="gemini_pro", model="gemini-3-pro-preview", role="search"
        )
        tools = _get_tools_for_model(cfg)
        assert mock_google_search in tools
        assert mock_extract in tools
        assert len(tools) == 2

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_grok_model_gets_only_extract(
        self, mock_extract: MagicMock, mock_google_search: MagicMock
    ) -> None:
        from magi_system.search.search_fanout import _get_tools_for_model

        cfg = ModelConfig(name="grok", model="xai/grok-4-1-fast", role="search")
        tools = _get_tools_for_model(cfg)
        assert mock_extract in tools
        assert mock_google_search not in tools
        assert len(tools) == 1

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_gpt_model_gets_only_extract(
        self, mock_extract: MagicMock, mock_google_search: MagicMock
    ) -> None:
        from magi_system.search.search_fanout import _get_tools_for_model

        cfg = ModelConfig(name="gpt", model="openai/gpt-5.1", role="search")
        tools = _get_tools_for_model(cfg)
        assert mock_extract in tools
        assert mock_google_search not in tools
        assert len(tools) == 1

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_claude_model_gets_only_extract(
        self, mock_extract: MagicMock, mock_google_search: MagicMock
    ) -> None:
        from magi_system.search.search_fanout import _get_tools_for_model

        cfg = ModelConfig(
            name="claude", model="anthropic/claude-opus-4-6", role="search"
        )
        tools = _get_tools_for_model(cfg)
        assert mock_extract in tools
        assert mock_google_search not in tools
        assert len(tools) == 1

    @patch("magi_system.search.search_fanout.google_search")
    @patch("magi_system.search.search_fanout.extract_page_content")
    def test_gemini_name_case_insensitive_match(
        self, mock_extract: MagicMock, mock_google_search: MagicMock
    ) -> None:
        from magi_system.search.search_fanout import _get_tools_for_model

        cfg = ModelConfig(name="Gemini_Flash", model="gemini-2.0-flash", role="search")
        tools = _get_tools_for_model(cfg)
        assert mock_google_search in tools
        assert len(tools) == 2


class TestGetModelForSearch:
    """Tests for _get_model_for_search function."""

    def test_grok_model_returns_litellm_with_extra_body(self) -> None:
        from google.adk.models.lite_llm import LiteLlm

        from magi_system.search.search_fanout import _get_model_for_search

        cfg = ModelConfig(
            name="grok",
            model=LiteLlm(model="xai/grok-4-1-fast"),
            role="search",
        )
        result = _get_model_for_search(cfg)
        assert isinstance(result, LiteLlm)

    def test_grok_model_has_search_parameters(self) -> None:
        from google.adk.models.lite_llm import LiteLlm

        from magi_system.search.search_fanout import _get_model_for_search

        cfg = ModelConfig(
            name="grok",
            model=LiteLlm(model="xai/grok-4-1-fast"),
            role="search",
        )
        result = _get_model_for_search(cfg)
        assert isinstance(result, LiteLlm)
        # Verify extra_body with search_parameters is in _additional_args
        assert result._additional_args["extra_body"] == {
            "search_parameters": {"mode": "auto"}
        }

    def test_gemini_model_returns_string_as_is(self) -> None:
        from magi_system.search.search_fanout import _get_model_for_search

        cfg = ModelConfig(
            name="gemini_pro",
            model="gemini-3-pro-preview",
            role="search",
        )
        result = _get_model_for_search(cfg)
        assert result == "gemini-3-pro-preview"

    def test_gpt_litellm_model_returns_as_is(self) -> None:
        from google.adk.models.lite_llm import LiteLlm

        from magi_system.search.search_fanout import _get_model_for_search

        cfg = ModelConfig(
            name="gpt",
            model=LiteLlm(model="openai/gpt-5.1"),
            role="search",
        )
        result = _get_model_for_search(cfg)
        assert isinstance(result, LiteLlm)

    def test_claude_litellm_model_returns_as_is(self) -> None:
        from google.adk.models.lite_llm import LiteLlm

        from magi_system.search.search_fanout import _get_model_for_search

        cfg = ModelConfig(
            name="claude",
            model=LiteLlm(model="anthropic/claude-opus-4-6"),
            role="search",
        )
        result = _get_model_for_search(cfg)
        assert isinstance(result, LiteLlm)


class TestSearchInstructionTemplate:
    """Tests for SEARCH_INSTRUCTION_TEMPLATE."""

    def test_template_exists(self) -> None:
        from magi_system.search.search_fanout import SEARCH_INSTRUCTION_TEMPLATE

        assert isinstance(SEARCH_INSTRUCTION_TEMPLATE, str)

    def test_template_has_research_brief_placeholder(self) -> None:
        from magi_system.search.search_fanout import SEARCH_INSTRUCTION_TEMPLATE

        assert "{research_brief}" in SEARCH_INSTRUCTION_TEMPLATE

    def test_template_has_sub_question_placeholder(self) -> None:
        from magi_system.search.search_fanout import SEARCH_INSTRUCTION_TEMPLATE

        assert "{sub_question}" in SEARCH_INSTRUCTION_TEMPLATE

    def test_template_has_claim_format(self) -> None:
        from magi_system.search.search_fanout import SEARCH_INSTRUCTION_TEMPLATE

        assert "CLAIM:" in SEARCH_INSTRUCTION_TEMPLATE
        assert "SOURCE:" in SEARCH_INSTRUCTION_TEMPLATE
        assert "CONFIDENCE:" in SEARCH_INSTRUCTION_TEMPLATE
        assert "EVIDENCE:" in SEARCH_INSTRUCTION_TEMPLATE

    def test_template_sub_question_is_format_replaceable(self) -> None:
        from magi_system.search.search_fanout import SEARCH_INSTRUCTION_TEMPLATE

        result = SEARCH_INSTRUCTION_TEMPLATE.format(
            research_brief="{research_brief}",
            sub_question="What is quantum computing?",
        )
        assert "What is quantum computing?" in result
        # research_brief should remain as ADK template placeholder
        assert "{research_brief}" in result
