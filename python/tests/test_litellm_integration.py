"""Tests for MAGI-012: Verify LiteLLM model integration with ADK.

This module contains two categories of tests:

1. Structural tests (no API keys needed) — verify that LiteLlm can be
   imported, instantiated for each provider, and used with ADK Agent
   alongside tool functions like extract_page_content.

2. Integration tests (require real API keys) — marked with
   @pytest.mark.integration and skipped by default when the corresponding
   environment variable is not set.

Known limitations:
- xAI's new search tools API (web_search / x_search tool types) is not
  yet supported by LiteLLM; use legacy search_parameters extra_body instead.
- OpenAI's web_search_options via extra_body may not be supported in all
  LiteLLM versions.
- The ADK LiteLlm wrapper stores extra_body in _additional_args, not as
  a top-level attribute.
"""

import inspect
import os

import pytest
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from magi_system.config import ModelConfig, load_config
from magi_system.tools.content_extraction import extract_page_content


# ---------------------------------------------------------------------------
# Structural tests — always run, no API keys required
# ---------------------------------------------------------------------------


class TestLiteLlmImport:
    """Verify LiteLlm can be imported from the expected module path."""

    def test_litellm_importable(self) -> None:
        from google.adk.models.lite_llm import LiteLlm as Imported

        assert Imported is LiteLlm

    def test_litellm_is_class(self) -> None:
        assert inspect.isclass(LiteLlm)


class TestLiteLlmInstantiation:
    """Verify LiteLlm instances can be created for each provider."""

    def test_openai_provider(self) -> None:
        llm = LiteLlm(model="openai/gpt-4o-mini")
        assert llm.model == "openai/gpt-4o-mini"

    def test_xai_provider(self) -> None:
        llm = LiteLlm(model="xai/grok-3-mini-fast")
        assert llm.model == "xai/grok-3-mini-fast"

    def test_anthropic_provider(self) -> None:
        llm = LiteLlm(model="anthropic/claude-sonnet-4-20250514")
        assert llm.model == "anthropic/claude-sonnet-4-20250514"

    def test_extra_body_accepted(self) -> None:
        llm = LiteLlm(
            model="xai/grok-3-mini-fast",
            extra_body={"search_parameters": {"mode": "auto"}},
        )
        assert llm._additional_args["extra_body"] == {
            "search_parameters": {"mode": "auto"}
        }

    def test_extra_body_openai_search_options(self) -> None:
        llm = LiteLlm(
            model="openai/gpt-4o-mini",
            extra_body={
                "web_search_options": {"search_context_size": "medium"}
            },
        )
        assert "web_search_options" in llm._additional_args["extra_body"]


class TestExtractPageContentToolCompatibility:
    """Verify extract_page_content is compatible with ADK tool requirements."""

    def test_is_callable(self) -> None:
        assert callable(extract_page_content)

    def test_has_type_hints(self) -> None:
        sig = inspect.signature(extract_page_content)
        for param in sig.parameters.values():
            assert param.annotation is not inspect.Parameter.empty, (
                f"Parameter '{param.name}' missing type hint"
            )

    def test_has_docstring(self) -> None:
        assert extract_page_content.__doc__ is not None
        assert len(extract_page_content.__doc__) > 0

    def test_has_return_annotation(self) -> None:
        sig = inspect.signature(extract_page_content)
        assert sig.return_annotation is not inspect.Signature.empty


class TestAgentWithLiteLlm:
    """Verify ADK Agent can be instantiated with LiteLlm + tool functions."""

    def test_agent_with_openai_litellm(self) -> None:
        agent = Agent(
            name="test_openai",
            model=LiteLlm(model="openai/gpt-4o-mini"),
            description="Test agent",
            instruction="You are a test agent.",
            tools=[extract_page_content],
        )
        assert agent.name == "test_openai"
        assert isinstance(agent.model, LiteLlm)
        assert extract_page_content in agent.tools

    def test_agent_with_xai_litellm(self) -> None:
        agent = Agent(
            name="test_grok",
            model=LiteLlm(
                model="xai/grok-3-mini-fast",
                extra_body={"search_parameters": {"mode": "auto"}},
            ),
            description="Test agent",
            instruction="You are a test agent.",
            tools=[extract_page_content],
        )
        assert agent.name == "test_grok"
        assert isinstance(agent.model, LiteLlm)

    def test_agent_with_anthropic_litellm(self) -> None:
        agent = Agent(
            name="test_claude",
            model=LiteLlm(model="anthropic/claude-sonnet-4-20250514"),
            description="Test agent",
            instruction="You are a test agent.",
            tools=[extract_page_content],
        )
        assert agent.name == "test_claude"
        assert isinstance(agent.model, LiteLlm)


class TestModelConfigStrings:
    """Verify ModelConfig from config.py has correct model strings."""

    def test_orchestrator_uses_gemini(self) -> None:
        config = load_config()
        assert config.orchestrator.model == "gemini-3-pro-preview"

    def test_synthesis_uses_anthropic_litellm(self) -> None:
        config = load_config()
        assert isinstance(config.synthesis.model, LiteLlm)
        assert "anthropic/" in config.synthesis.model.model

    def test_search_gpt_model_string(self) -> None:
        config = load_config()
        search_by_name = {m.name: m for m in config.search_models}
        gpt = search_by_name["gpt"]
        assert isinstance(gpt.model, LiteLlm)
        assert gpt.model.model.startswith("openai/")

    def test_search_grok_model_string(self) -> None:
        config = load_config()
        search_by_name = {m.name: m for m in config.search_models}
        grok = search_by_name["grok"]
        assert isinstance(grok.model, LiteLlm)
        assert grok.model.model.startswith("xai/")

    def test_reflection_claude_model_string(self) -> None:
        config = load_config()
        reflection_by_name = {m.name: m for m in config.reflection_models}
        claude = reflection_by_name["claude"]
        assert isinstance(claude.model, LiteLlm)
        assert claude.model.model.startswith("anthropic/")

    def test_all_litellm_models_have_provider_prefix(self) -> None:
        """Every LiteLlm model string should follow 'provider/model' format."""
        config = load_config()
        all_models = (
            [config.orchestrator, config.synthesis]
            + config.search_models
            + config.reflection_models
        )
        for mc in all_models:
            if isinstance(mc.model, LiteLlm):
                assert "/" in mc.model.model, (
                    f"LiteLlm model '{mc.model.model}' missing provider prefix"
                )


# ---------------------------------------------------------------------------
# Integration tests — require API keys, skipped by default
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests for OpenAI via LiteLlm (requires OPENAI_API_KEY)."""

    pytestmark = pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )

    def test_agent_instantiation(self) -> None:
        agent = Agent(
            name="openai_integration",
            model=LiteLlm(model="openai/gpt-4o-mini"),
            description="OpenAI integration test agent.",
            instruction="Respond with a single word: hello.",
            tools=[extract_page_content],
        )
        assert agent is not None


@pytest.mark.integration
class TestXAIIntegration:
    """Integration tests for xAI/Grok via LiteLlm (requires XAI_API_KEY)."""

    pytestmark = pytest.mark.skipif(
        not os.environ.get("XAI_API_KEY"),
        reason="XAI_API_KEY not set",
    )

    def test_agent_instantiation_with_search_params(self) -> None:
        agent = Agent(
            name="grok_integration",
            model=LiteLlm(
                model="xai/grok-3-mini-fast",
                extra_body={"search_parameters": {"mode": "auto"}},
            ),
            description="Grok integration test agent.",
            instruction="Respond with a single word: hello.",
            tools=[extract_page_content],
        )
        assert agent is not None


@pytest.mark.integration
class TestAnthropicIntegration:
    """Integration tests for Anthropic via LiteLlm (requires ANTHROPIC_API_KEY)."""

    pytestmark = pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )

    def test_agent_instantiation(self) -> None:
        agent = Agent(
            name="claude_integration",
            model=LiteLlm(model="anthropic/claude-sonnet-4-20250514"),
            description="Claude integration test agent.",
            instruction="Respond with a single word: hello.",
            tools=[extract_page_content],
        )
        assert agent is not None
