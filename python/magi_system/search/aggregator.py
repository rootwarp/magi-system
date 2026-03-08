"""Search results aggregator for the MAGI system.

Collects all sq{N}_{model} search result keys from session state
and aggregates them into a single all_search_results key.
"""

from __future__ import annotations

import re
from typing import AsyncIterator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types as genai_types

_SQ_KEY_PATTERN = re.compile(r"^sq(\d+)_(.+)$")


class SearchResultsAggregator(BaseAgent):
    """Aggregates search results from multiple models and sub-questions.

    Reads all state keys matching the pattern sq{N}_{model} (e.g.,
    sq1_gemini, sq2_gpt) and concatenates them with clear headers
    into a single all_search_results state key.
    """

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncIterator[Event]:
        state = ctx.session.state
        results = []

        for key in sorted(state.keys()):
            match = _SQ_KEY_PATTERN.match(key)
            if match and state[key]:
                sq_num = match.group(1)
                model_name = match.group(2)
                results.append(
                    f"## Sub-Question {sq_num} — {model_name}\n\n{state[key]}\n"
                )

        aggregated = (
            "\n---\n\n".join(results) if results else "No search results available."
        )
        state["all_search_results"] = aggregated

        yield genai_types.Content(
            role="model",
            parts=[genai_types.Part(text=f"Aggregated {len(results)} search results.")],
        )
