"""Tests for MAGI-010: Sub-question parser in search_fanout module."""

import pytest

from magi_system.search.search_fanout import _parse_sub_questions


class TestParseSubQuestionsBasic:
    """Tests for basic sub-question parsing."""

    def test_parses_standard_format(self) -> None:
        plan = (
            "SUB_QUESTIONS:\n"
            "SQ1: What are the current market trends for AI chips?\n"
            "PRIORITY: high\n"
            "DEPENDS_ON: none\n"
            "\n"
            "SQ2: How do NVIDIA's latest offerings compare to AMD's?\n"
            "PRIORITY: high\n"
            "DEPENDS_ON: SQ1\n"
        )
        result = _parse_sub_questions(plan)
        assert result == [
            "What are the current market trends for AI chips?",
            "How do NVIDIA's latest offerings compare to AMD's?",
        ]

    def test_parses_full_example(self) -> None:
        plan = (
            "COMPLEXITY: medium\n"
            "\n"
            "SUB_QUESTIONS:\n"
            "SQ1: What are the current market trends for AI chips?\n"
            "PRIORITY: high\n"
            "DEPENDS_ON: none\n"
            "\n"
            "SQ2: How do NVIDIA's latest offerings compare to AMD's?\n"
            "PRIORITY: high\n"
            "DEPENDS_ON: SQ1\n"
            "\n"
            "SQ3: What is the total addressable market for AI accelerators?\n"
            "PRIORITY: medium\n"
            "DEPENDS_ON: none\n"
        )
        result = _parse_sub_questions(plan)
        assert result == [
            "What are the current market trends for AI chips?",
            "How do NVIDIA's latest offerings compare to AMD's?",
            "What is the total addressable market for AI accelerators?",
        ]

    def test_single_sub_question(self) -> None:
        plan = "SQ1: What is Python?\n"
        result = _parse_sub_questions(plan)
        assert result == ["What is Python?"]

    def test_returns_list_type(self) -> None:
        plan = "SQ1: A question?\n"
        result = _parse_sub_questions(plan)
        assert isinstance(result, list)


class TestParseSubQuestionsEdgeCases:
    """Tests for edge cases in sub-question parsing."""

    def test_empty_string(self) -> None:
        assert _parse_sub_questions("") == []

    def test_no_sub_questions(self) -> None:
        plan = "COMPLEXITY: medium\nSome random text\n"
        assert _parse_sub_questions(plan) == []

    def test_whitespace_only(self) -> None:
        assert _parse_sub_questions("   \n  \n  ") == []

    def test_leading_trailing_whitespace_on_lines(self) -> None:
        plan = "  SQ1: What is AI?  \n  SQ2: What is ML?  \n"
        result = _parse_sub_questions(plan)
        assert result == ["What is AI?", "What is ML?"]

    def test_extra_whitespace_in_question(self) -> None:
        plan = "SQ1:   What is AI?  \n"
        result = _parse_sub_questions(plan)
        assert result == ["What is AI?"]

    def test_numbering_gaps(self) -> None:
        plan = "SQ1: First question?\nSQ3: Third question?\n"
        result = _parse_sub_questions(plan)
        assert result == ["First question?", "Third question?"]

    def test_malformed_lines_ignored(self) -> None:
        plan = (
            "SQ1: Valid question?\n"
            "This is not a sub-question\n"
            "SQ2: Another valid question?\n"
        )
        result = _parse_sub_questions(plan)
        assert result == ["Valid question?", "Another valid question?"]

    def test_priority_and_depends_not_included(self) -> None:
        plan = (
            "SQ1: A question?\n"
            "PRIORITY: high\n"
            "DEPENDS_ON: none\n"
        )
        result = _parse_sub_questions(plan)
        assert len(result) == 1
        assert result[0] == "A question?"

    def test_sq_with_space_before_number(self) -> None:
        plan = "SQ 1: What is AI?\nSQ 2: What is ML?\n"
        result = _parse_sub_questions(plan)
        assert result == ["What is AI?", "What is ML?"]

    def test_sq_with_dash_separator(self) -> None:
        plan = "SQ1 - What is AI?\nSQ2 - What is ML?\n"
        result = _parse_sub_questions(plan)
        assert result == ["What is AI?", "What is ML?"]

    def test_mixed_formats(self) -> None:
        plan = (
            "SQ1: First question?\n"
            "SQ 2: Second question?\n"
            "SQ3 - Third question?\n"
        )
        result = _parse_sub_questions(plan)
        assert result == [
            "First question?",
            "Second question?",
            "Third question?",
        ]

    def test_large_sq_numbers(self) -> None:
        plan = "SQ10: Tenth question?\nSQ15: Fifteenth question?\n"
        result = _parse_sub_questions(plan)
        assert result == ["Tenth question?", "Fifteenth question?"]

    def test_empty_question_text_ignored(self) -> None:
        plan = "SQ1: \nSQ2: Valid question?\n"
        result = _parse_sub_questions(plan)
        assert result == ["Valid question?"]

    def test_does_not_raise_on_bad_input(self) -> None:
        # Should never raise, just return empty list
        assert _parse_sub_questions("garbage data !@#$%") == []

    def test_multiline_with_blank_lines(self) -> None:
        plan = (
            "\n\n"
            "SQ1: First?\n"
            "\n\n"
            "SQ2: Second?\n"
            "\n\n"
        )
        result = _parse_sub_questions(plan)
        assert result == ["First?", "Second?"]

    def test_preserves_order(self) -> None:
        plan = "SQ3: Third?\nSQ1: First?\nSQ2: Second?\n"
        result = _parse_sub_questions(plan)
        assert result == ["Third?", "First?", "Second?"]

    def test_colon_in_question_text(self) -> None:
        plan = "SQ1: What is the ratio: 1:2 or 1:3?\n"
        result = _parse_sub_questions(plan)
        assert result == ["What is the ratio: 1:2 or 1:3?"]
