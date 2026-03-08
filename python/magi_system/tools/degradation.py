"""Graceful degradation utilities for handling provider failures."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineWarnings:
    """Collects degradation notices during pipeline execution."""

    warnings: list[str] = field(default_factory=list)
    unavailable_models: list[str] = field(default_factory=list)

    def add_model_failure(self, model_name: str, error: str):
        """Record a model failure."""
        self.unavailable_models.append(model_name)
        msg = f"Model {model_name} unavailable: {error}"
        self.warnings.append(msg)
        logger.warning(msg)

    def add_warning(self, message: str):
        """Record a general warning."""
        self.warnings.append(message)
        logger.warning(message)

    @property
    def has_degradation(self) -> bool:
        return len(self.unavailable_models) > 0

    def format_for_report(self) -> str:
        """Format warnings for inclusion in final report."""
        if not self.warnings:
            return ""
        lines = ["## Pipeline Warnings", ""]
        for w in self.warnings:
            lines.append(f"- {w}")
        if self.unavailable_models:
            lines.append("")
            lines.append(f"Models unavailable: {', '.join(self.unavailable_models)}")
            lines.append("Confidence scores may be affected by reduced model diversity.")
        return "\n".join(lines)
