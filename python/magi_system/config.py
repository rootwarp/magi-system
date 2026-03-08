"""Configuration module for the MAGI pipeline system.

Defines model configurations and pipeline parameters used across
the multi-agent research and synthesis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from google.adk.models.lite_llm import LiteLlm


@dataclass
class ModelConfig:
    """Configuration for a single model in the pipeline.

    Attributes:
        name: Human-readable identifier for this model configuration.
        model: Model identifier string or LiteLlm wrapper instance.
        enabled: Whether this model is active in the pipeline.
        role: The role this model plays (e.g., "orchestrator", "search").
    """

    name: str
    model: Union[str, LiteLlm]
    role: str
    enabled: bool = True


@dataclass
class PipelineConfig:
    """Configuration for the full MAGI research pipeline.

    Attributes:
        orchestrator: Model config for the pipeline orchestrator.
        synthesis: Model config for the final synthesis step.
        search_models: List of model configs for parallel search agents.
        reflection_models: List of model configs for reflection agents.
        max_research_iterations: Maximum number of research loop iterations.
        max_discussion_rounds: Maximum rounds of multi-agent discussion.
        max_sub_questions: Maximum sub-questions to generate from a query.
        divergence_threshold: Threshold for detecting model disagreement.
        min_models_for_consensus: Minimum models that must agree for consensus.
        max_searches_per_agent: Maximum search calls each agent can make.
        api_timeout_seconds: Timeout in seconds for external API calls.
        max_retries: Maximum number of retry attempts for transient failures.
        retry_base_delay: Base delay in seconds for exponential backoff.
    """

    orchestrator: ModelConfig
    synthesis: ModelConfig
    search_models: list[ModelConfig]
    reflection_models: list[ModelConfig]
    max_research_iterations: int = 3
    max_discussion_rounds: int = 3
    max_sub_questions: int = 15
    divergence_threshold: float = 0.7
    min_models_for_consensus: int = 2
    max_searches_per_agent: int = 5
    api_timeout_seconds: int = 30
    max_retries: int = 3
    retry_base_delay: float = 1.0


def load_config() -> PipelineConfig:
    """Load the default pipeline configuration.

    Returns:
        PipelineConfig with default model assignments and parameters.
    """
    return PipelineConfig(
        orchestrator=ModelConfig(
            name="gemini_pro",
            model="gemini-3-pro-preview",
            role="orchestrator",
        ),
        synthesis=ModelConfig(
            name="claude",
            model=LiteLlm(model="anthropic/claude-opus-4-6"),
            role="synthesis",
        ),
        search_models=[
            ModelConfig(
                name="gemini_pro",
                model="gemini-3-pro-preview",
                role="search",
            ),
            ModelConfig(
                name="gpt",
                model=LiteLlm(model="openai/gpt-5.1"),
                role="search",
            ),
            ModelConfig(
                name="grok",
                model=LiteLlm(model="xai/grok-4-1-fast"),
                role="search",
            ),
        ],
        reflection_models=[
            ModelConfig(
                name="gemini_pro",
                model="gemini-3-pro-preview",
                role="reflection",
            ),
            ModelConfig(
                name="claude",
                model=LiteLlm(model="anthropic/claude-opus-4-6"),
                role="reflection",
            ),
        ],
    )
