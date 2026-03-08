"""Dual reflection agents for evaluating search coverage (MAGI-020).

Two reflection agents (Gemini + Claude) run in parallel, each independently
evaluating search coverage against the research plan.
"""

from google.adk.agents import LlmAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm

REFLECTION_INSTRUCTION = """\
You are a research quality evaluator. Your task is to assess whether the \
search results adequately cover the research plan.

IMPORTANT: Include all reasoning and analysis inline in your response. \
Do not rely on hidden reasoning or chain-of-thought that may be discarded.

Research Plan:
{research_plan}

Search Results:
{all_search_results}

Evaluate the search results against the research plan by performing the \
following analysis:

## Coverage Assessment

For each sub-question in the research plan, assess coverage:
- COMPLETE: Sub-question is fully answered with high-quality, corroborated \
evidence from multiple sources.
- PARTIAL: Some relevant information found but significant gaps remain, or \
evidence quality is insufficient.
- MISSING: No meaningful information found for this sub-question.

## Gap Analysis

Identify gaps in coverage and assign severity:
- CRITICAL: Essential information missing that would make the final report \
unreliable or incomplete. Must be addressed.
- IMPORTANT: Significant gaps that would weaken the report but not \
invalidate it. Should be addressed if possible.
- MINOR: Nice-to-have information that would enrich the report but is not \
essential.

## Contradiction Detection

Examine the search results for contradictions across different model outputs \
or sources. For each contradiction found, note:
- The conflicting claims
- The sources involved
- Which claim appears more credible and why

## Final Recommendation

Based on your analysis, provide:
LOOP_BACK: [yes|no]
REASON: [Brief explanation of why another search iteration is or is not needed]
QUALITY_SCORE: [1-10] where 1 = completely inadequate and 10 = comprehensive \
coverage with high confidence\
"""

def create_dual_reflection() -> ParallelAgent:
    """Create a fresh dual reflection ParallelAgent.

    Returns a new instance each time because ADK agents track parent
    references and cannot be reused across multiple parent agents.
    """
    reflection_gemini = LlmAgent(
        name="reflection_gemini",
        model="gemini-3-pro-preview",
        instruction=REFLECTION_INSTRUCTION,
        output_key="reflection_gemini",
        description="Gemini reflection agent evaluating search coverage.",
    )

    reflection_claude = LlmAgent(
        name="reflection_claude",
        model=LiteLlm(model="anthropic/claude-opus-4-6"),
        instruction=REFLECTION_INSTRUCTION,
        output_key="reflection_claude",
        description="Claude reflection agent evaluating search coverage.",
    )

    return ParallelAgent(
        name="dual_reflection",
        sub_agents=[reflection_gemini, reflection_claude],
        description="Runs Gemini and Claude reflection agents in parallel.",
    )


dual_reflection = create_dual_reflection()
