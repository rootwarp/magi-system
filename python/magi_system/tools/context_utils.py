"""Context management utilities to prevent token overflow."""

from __future__ import annotations


def truncate_text(
    text: str, max_chars: int = 50000, preserve_start: int = 5000
) -> str:
    """Truncate text to max_chars while preserving key information.

    Keeps the first preserve_start chars and the last portion up to max_chars.

    Args:
        text: The text to potentially truncate.
        max_chars: Maximum allowed character count.
        preserve_start: Number of characters to preserve from the beginning.

    Returns:
        The original text if within limits, or a truncated version with
        a marker indicating where content was removed.
    """
    if len(text) <= max_chars:
        return text
    separator = "\n\n... [TRUNCATED — content exceeded context limit] ...\n\n"
    # Clamp preserve_start so it doesn't exceed available space
    effective_start = min(preserve_start, max_chars - len(separator))
    if effective_start < 0:
        # max_chars is smaller than separator itself; just hard-truncate
        return text[:max_chars]
    end_chars = max_chars - effective_start - len(separator)
    if end_chars <= 0:
        return text[:effective_start] + separator
    return text[:effective_start] + separator + text[-end_chars:]


def summarize_search_results(results: str, max_chars: int = 50000) -> str:
    """Summarize/truncate search results to fit within context limits.

    Args:
        results: Raw search results text.
        max_chars: Maximum allowed character count.

    Returns:
        Results truncated to fit within the specified limit.
    """
    return truncate_text(results, max_chars)
