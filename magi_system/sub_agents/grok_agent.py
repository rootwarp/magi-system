"""Grok agent specialized for logical reasoning and decision-making."""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

from ..tools.reasoning_tools import chain_of_thought, compare_options


load_dotenv()

agent = Agent(
    name="grok_agent",
    model=LiteLlm(
        model="xai/grok-4-1-fast-reasoning",
        # xAI Search Tools options:
        # 1. Legacy live_search (deprecated Dec 15, 2025):
        extra_body={"search_parameters": {"mode": "auto"}},
        # 2. New search tools API (not yet supported - API returns error):
        #    extra_body={"tools": [{"type": "web_search"}, {"type": "x_search"}]},
        # See: https://docs.x.ai/docs/guides/tools/search-tools
    ),
    description="Reasoning specialist providing one perspective in the MAGI voting system.",
    instruction="""You are a reasoning specialist powered by Grok, participating in a multi-agent voting system.

Your role is to provide your BEST response to the user's query from a LOGICAL REASONING perspective.
Even if the task is not primarily reasoning-focused, apply your unique capabilities:
- Break down the problem systematically
- Apply structured analytical frameworks
- Provide well-reasoned recommendations

When responding:
1. Use chain_of_thought for complex problem analysis when appropriate
2. Use compare_options when evaluating alternatives
3. Show your reasoning process step by step
4. Provide clear, actionable recommendations

IMPORTANT: Provide a complete, standalone response. Other agents will also respond,
and a synthesizer will combine all perspectives. Include your reasoning and confidence level.""",
    tools=[chain_of_thought, compare_options],
    output_key="grok_result",
)
