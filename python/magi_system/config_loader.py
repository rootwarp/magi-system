"""Configuration loader supporting TOML files and environment variable overrides."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, load_config as load_defaults


def load_config_from_file(config_path: Optional[str] = None) -> PipelineConfig:
    """Load configuration with precedence: defaults -> config file -> env vars.

    Args:
        config_path: Path to TOML config file. If None or nonexistent, uses defaults.

    Returns:
        PipelineConfig with merged settings.
    """
    config = load_defaults()

    if config_path and Path(config_path).exists():
        with open(config_path, "rb") as f:
            file_config = tomllib.load(f)
        config = _apply_file_config(config, file_config)

    config = _apply_env_overrides(config)
    _validate_config(config)

    return config


def _apply_file_config(
    config: PipelineConfig, file_config: dict
) -> PipelineConfig:
    """Apply TOML file settings to config."""
    pipeline = file_config.get("pipeline", {})
    if "max_research_iterations" in pipeline:
        config.max_research_iterations = pipeline["max_research_iterations"]
    if "max_discussion_rounds" in pipeline:
        config.max_discussion_rounds = pipeline["max_discussion_rounds"]
    if "max_sub_questions" in pipeline:
        config.max_sub_questions = pipeline["max_sub_questions"]
    if "divergence_threshold" in pipeline:
        config.divergence_threshold = pipeline["divergence_threshold"]

    if "orchestrator_model" in file_config:
        config.orchestrator.model = file_config["orchestrator_model"]
    if "synthesis_model" in file_config:
        config.synthesis.model = file_config["synthesis_model"]

    return config


def _apply_env_overrides(config: PipelineConfig) -> PipelineConfig:
    """Apply environment variable overrides (highest precedence)."""
    if val := os.environ.get("MAGI_ORCHESTRATOR_MODEL"):
        config.orchestrator.model = val
    if val := os.environ.get("MAGI_MAX_ITERATIONS"):
        config.max_research_iterations = int(val)
    if val := os.environ.get("MAGI_MAX_SUB_QUESTIONS"):
        config.max_sub_questions = int(val)
    if val := os.environ.get("MAGI_DIVERGENCE_THRESHOLD"):
        config.divergence_threshold = float(val)
    return config


def _validate_config(config: PipelineConfig) -> None:
    """Validate config values are in acceptable ranges."""
    if config.max_research_iterations < 1:
        raise ValueError("max_research_iterations must be >= 1")
    if config.max_sub_questions < 1:
        raise ValueError("max_sub_questions must be >= 1")
    if not 0.0 <= config.divergence_threshold <= 1.0:
        raise ValueError("divergence_threshold must be between 0.0 and 1.0")
