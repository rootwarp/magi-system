"""Tests for MAGI-021: Divergence detector configuration and structure."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


def _mock_google_adk():
    """Set up mocks for google.adk modules so imports resolve without credentials."""
    mock_llm_agent = MagicMock()

    adk_modules = {
        "google": MagicMock(),
        "google.adk": MagicMock(),
        "google.adk.agents": MagicMock(LlmAgent=mock_llm_agent),
        "google.adk.models": MagicMock(),
        "google.adk.models.lite_llm": MagicMock(),
        "google.adk.tools": MagicMock(),
        "google.adk.tools.agent_tool": MagicMock(),
        "google.adk.tools.exit_loop_tool": MagicMock(),
        "google.adk.tools.google_search": MagicMock(),
    }
    saved = {}
    for name, mock in adk_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock
    return saved, mock_llm_agent


def _restore_modules(saved):
    """Restore original sys.modules entries."""
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture(autouse=True)
def mock_adk():
    """Mock google.adk for all tests in this module, then clean up."""
    saved, mock_llm_agent = _mock_google_adk()
    # Clear cached magi_system modules so they re-import with mocks
    magi_keys = [k for k in sys.modules if k.startswith("magi_system")]
    saved_magi = {k: sys.modules.pop(k) for k in magi_keys}
    yield mock_llm_agent
    # Restore everything
    for k in list(sys.modules):
        if k.startswith("magi_system"):
            del sys.modules[k]
    sys.modules.update(saved_magi)
    _restore_modules(saved)


class TestDivergenceDetectorImport:
    """Verify divergence_detector is importable."""

    def test_import_from_consensus_module(self):
        from magi_system.consensus import divergence_detector

        assert divergence_detector is not None

    def test_import_from_divergence_module(self):
        from magi_system.consensus.divergence import divergence_detector

        assert divergence_detector is not None

    def test_module_importable(self):
        mod = importlib.import_module("magi_system.consensus.divergence")
        assert mod is not None


class TestDivergenceDetectorConfiguration:
    """Verify divergence_detector is configured correctly per MAGI-021 spec."""

    def test_agent_created_with_llm_agent(self, mock_adk):
        from magi_system.consensus.divergence import divergence_detector  # noqa: F811

        calls = [
            c
            for c in mock_adk.call_args_list
            if c[1].get("name") == "divergence_detector"
        ]
        assert len(calls) == 1

    def test_agent_model_is_gemini_3_pro(self, mock_adk):
        from magi_system.consensus.divergence import divergence_detector  # noqa: F811

        call_kwargs = mock_adk.call_args[1]
        assert call_kwargs["model"] == "gemini-3-pro-preview"

    def test_agent_output_key_is_divergence_map(self, mock_adk):
        from magi_system.consensus.divergence import divergence_detector  # noqa: F811

        call_kwargs = mock_adk.call_args[1]
        assert call_kwargs["output_key"] == "divergence_map"

    def test_agent_has_description(self, mock_adk):
        from magi_system.consensus.divergence import divergence_detector  # noqa: F811

        call_kwargs = mock_adk.call_args[1]
        assert "description" in call_kwargs
        assert len(call_kwargs["description"]) > 0


class TestDivergenceDetectorInstruction:
    """Verify the instruction template contains required elements."""

    def _get_instruction(self, mock_adk):
        from magi_system.consensus.divergence import divergence_detector  # noqa: F811

        return mock_adk.call_args[1]["instruction"]

    def test_instruction_contains_research_plan(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "{research_plan}" in instruction

    def test_instruction_contains_reflection_gemini(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "{reflection_gemini}" in instruction

    def test_instruction_contains_reflection_claude(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "{reflection_claude}" in instruction

    def test_instruction_contains_all_search_results(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "{all_search_results}" in instruction

    def test_instruction_mentions_agreement_zone(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "AGREEMENT ZONE" in instruction or "AGREEMENT_ZONE" in instruction

    def test_instruction_mentions_resolved_by_vote(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert (
            "RESOLVED BY VOTE" in instruction or "RESOLVED_BY_VOTE" in instruction
        )

    def test_instruction_mentions_needs_discussion(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert (
            "NEEDS DISCUSSION" in instruction or "NEEDS_DISCUSSION" in instruction
        )

    def test_instruction_mentions_overall_agreement_score(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "OVERALL_AGREEMENT_SCORE" in instruction

    def test_instruction_mentions_discussion_needed(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "DISCUSSION_NEEDED" in instruction

    def test_instruction_mentions_contradiction_type(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "CONTRADICTION" in instruction

    def test_instruction_mentions_magnitude_type(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "MAGNITUDE" in instruction

    def test_instruction_mentions_omission_type(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "OMISSION" in instruction

    def test_instruction_mentions_ranking_type(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "RANKING" in instruction

    def test_instruction_mentions_voting_majority(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "2/3" in instruction or "two-thirds" in instruction.lower()

    def test_instruction_mentions_severity_levels(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "HIGH" in instruction
        assert "MEDIUM" in instruction
        assert "LOW" in instruction

    def test_instruction_mentions_confidence_scores(self, mock_adk):
        instruction = self._get_instruction(mock_adk)
        assert "0.0" in instruction and "1.0" in instruction
