"""Shared test fixtures for the MAGI system test suite."""

import pytest

from magi_system.config import ModelConfig, PipelineConfig, load_config


@pytest.fixture
def default_config() -> PipelineConfig:
    """Return the default pipeline configuration."""
    return load_config()


@pytest.fixture
def sample_model_config() -> ModelConfig:
    """Return a simple ModelConfig for testing."""
    return ModelConfig(name="test", model="test-model", role="test")
