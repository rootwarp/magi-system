"""Tests for MAGI-002: config.py module."""

from dataclasses import fields
from unittest.mock import patch

import pytest


class TestModelConfig:
    """Tests for the ModelConfig dataclass."""

    def test_model_config_exists(self) -> None:
        from magi_system.config import ModelConfig

        assert ModelConfig is not None

    def test_model_config_has_required_fields(self) -> None:
        from magi_system.config import ModelConfig

        field_names = {f.name for f in fields(ModelConfig)}
        assert "name" in field_names
        assert "model" in field_names
        assert "enabled" in field_names
        assert "role" in field_names

    def test_model_config_enabled_defaults_true(self) -> None:
        from magi_system.config import ModelConfig

        config = ModelConfig(name="test", model="test-model", role="test")
        assert config.enabled is True

    def test_model_config_with_string_model(self) -> None:
        from magi_system.config import ModelConfig

        config = ModelConfig(
            name="test", model="some-model-id", role="tester"
        )
        assert config.name == "test"
        assert config.model == "some-model-id"
        assert config.role == "tester"
        assert config.enabled is True

    def test_model_config_with_litellm_model(self) -> None:
        from google.adk.models.lite_llm import LiteLlm

        from magi_system.config import ModelConfig

        llm = LiteLlm(model="openai/gpt-5.1")
        config = ModelConfig(name="gpt", model=llm, role="search")
        assert isinstance(config.model, LiteLlm)

    def test_model_config_enabled_override(self) -> None:
        from magi_system.config import ModelConfig

        config = ModelConfig(
            name="test", model="test-model", role="test", enabled=False
        )
        assert config.enabled is False


class TestPipelineConfig:
    """Tests for the PipelineConfig dataclass."""

    def test_pipeline_config_exists(self) -> None:
        from magi_system.config import PipelineConfig

        assert PipelineConfig is not None

    def test_pipeline_config_has_model_assignments(self) -> None:
        from magi_system.config import PipelineConfig

        field_names = {f.name for f in fields(PipelineConfig)}
        assert "orchestrator" in field_names
        assert "synthesis" in field_names
        assert "search_models" in field_names
        assert "reflection_models" in field_names

    def test_pipeline_config_has_pipeline_params(self) -> None:
        from magi_system.config import PipelineConfig

        field_names = {f.name for f in fields(PipelineConfig)}
        assert "max_research_iterations" in field_names
        assert "max_discussion_rounds" in field_names
        assert "max_sub_questions" in field_names

    def test_pipeline_config_has_consensus_thresholds(self) -> None:
        from magi_system.config import PipelineConfig

        field_names = {f.name for f in fields(PipelineConfig)}
        assert "divergence_threshold" in field_names
        assert "min_models_for_consensus" in field_names

    def test_pipeline_config_has_search_params(self) -> None:
        from magi_system.config import PipelineConfig

        field_names = {f.name for f in fields(PipelineConfig)}
        assert "max_searches_per_agent" in field_names

    def test_pipeline_config_default_pipeline_params(self) -> None:
        from magi_system.config import load_config

        config = load_config()
        assert config.max_research_iterations == 3
        assert config.max_discussion_rounds == 3
        assert config.max_sub_questions == 15

    def test_pipeline_config_default_consensus_thresholds(self) -> None:
        from magi_system.config import load_config

        config = load_config()
        assert config.divergence_threshold == 0.7
        assert config.min_models_for_consensus == 2

    def test_pipeline_config_default_search_params(self) -> None:
        from magi_system.config import load_config

        config = load_config()
        assert config.max_searches_per_agent == 5


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_returns_pipeline_config(self) -> None:
        from magi_system.config import PipelineConfig, load_config

        config = load_config()
        assert isinstance(config, PipelineConfig)

    def test_load_config_orchestrator_model(self) -> None:
        from magi_system.config import load_config

        config = load_config()
        assert config.orchestrator.name == "gemini_pro"
        assert config.orchestrator.role == "orchestrator"
        assert config.orchestrator.enabled is True

    def test_load_config_synthesis_model(self) -> None:
        from magi_system.config import load_config

        config = load_config()
        assert config.synthesis.name == "claude"
        assert config.synthesis.role == "synthesis"
        assert config.synthesis.enabled is True

    def test_load_config_search_models(self) -> None:
        from magi_system.config import load_config

        config = load_config()
        assert len(config.search_models) == 3
        model_names = [m.name for m in config.search_models]
        assert "gemini_pro" in model_names
        assert "gpt" in model_names
        assert "grok" in model_names
        for m in config.search_models:
            assert m.role == "search"
            assert m.enabled is True

    def test_load_config_reflection_models(self) -> None:
        from magi_system.config import load_config

        config = load_config()
        assert len(config.reflection_models) == 2
        model_names = [m.name for m in config.reflection_models]
        assert "gemini_pro" in model_names
        assert "claude" in model_names
        for m in config.reflection_models:
            assert m.role == "reflection"
            assert m.enabled is True

    def test_load_config_model_ids(self) -> None:
        """Verify the specific model IDs used in defaults."""
        from google.adk.models.lite_llm import LiteLlm

        from magi_system.config import load_config

        config = load_config()

        # Orchestrator uses plain string model
        assert config.orchestrator.model == "gemini-3-pro-preview"

        # Synthesis uses LiteLlm wrapper
        assert isinstance(config.synthesis.model, LiteLlm)

        # Search models: gemini is string, others are LiteLlm
        search_by_name = {m.name: m for m in config.search_models}
        assert search_by_name["gemini_pro"].model == "gemini-3-pro-preview"
        assert isinstance(search_by_name["gpt"].model, LiteLlm)
        assert isinstance(search_by_name["grok"].model, LiteLlm)

    def test_load_config_type_hints(self) -> None:
        """Verify ModelConfig and PipelineConfig are proper dataclasses."""
        import dataclasses

        from magi_system.config import ModelConfig, PipelineConfig

        assert dataclasses.is_dataclass(ModelConfig)
        assert dataclasses.is_dataclass(PipelineConfig)
