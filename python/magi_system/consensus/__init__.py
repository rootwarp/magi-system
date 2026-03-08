from __future__ import annotations

from typing import AsyncIterator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types as genai_types

from ..config import PipelineConfig
from .discussion import create_discussion_loop
from .divergence import divergence_detector


class ConsensusAgent(BaseAgent):
    """Orchestrates consensus: divergence detection followed by conditional discussion.

    Phase A always runs divergence detection with voting fast path.
    Phase B (discussion loop) only runs when divergence_map indicates
    DISCUSSION_NEEDED: yes.
    """

    config: PipelineConfig
    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncIterator[Event]:
        # Phase A: Always run divergence detection
        async for event in divergence_detector.run_async(ctx):
            yield event

        # Check if discussion is needed (case-insensitive)
        divergence_map = ctx.session.state.get("divergence_map", "")
        discussion_needed = "discussion_needed: yes" in divergence_map.lower()

        if discussion_needed:
            # Phase B: Run discussion loop
            discussion_loop = create_discussion_loop(self.config)
            async for event in discussion_loop.run_async(ctx):
                yield event

            yield genai_types.Content(
                role="model",
                parts=[genai_types.Part(text="Consensus reached via discussion.")],
            )
        else:
            # Fast path: voting resolved all divergences
            ctx.session.state["consensus_result"] = (
                "CONSENSUS REACHED VIA VOTING\n\n" + divergence_map
            )
            yield genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        text="Consensus reached via voting fast path."
                    )
                ],
            )


__all__ = ["ConsensusAgent", "create_discussion_loop", "divergence_detector"]
