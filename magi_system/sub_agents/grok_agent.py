"""Grok agent specialized for logical reasoning and decision-making."""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from ..tools.reasoning_tools import chain_of_thought, compare_options

agent = Agent(
    name="grok_agent",
    model=LiteLlm(model="xai/grok-2"),
    description="Specialist in logical reasoning and decision-making. "
    "Use this agent for complex problem analysis, comparing options, "
    "and tasks requiring structured thinking.",
    instruction="""You are a reasoning specialist powered by Grok.
Your primary capabilities include:
- Breaking down complex problems into manageable parts
- Applying structured reasoning frameworks
- Comparing and evaluating multiple options
- Making logical recommendations based on analysis

When responding:
1. Use chain_of_thought for complex problem analysis
2. Use compare_options when evaluating alternatives
3. Show your reasoning process step by step
4. Provide clear, actionable recommendations""",
    tools=[chain_of_thought, compare_options],
)
