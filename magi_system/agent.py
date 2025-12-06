"""MAGI System voting architecture using ADK Workflow Agents.

This is the main entry point for the MAGI multi-agent system.
All requests are processed by ALL specialists in parallel, then synthesized
into a unified response with disagreement analysis.
"""

from google.adk.agents import ParallelAgent, SequentialAgent
from dotenv import load_dotenv

from .sub_agents import gemini_agent, grok_agent, openai_agent, synthesizer_agent


load_dotenv()

# Step 1: ParallelAgent runs all three specialists concurrently
# Each specialist provides their perspective on the user's query
parallel_analysis = ParallelAgent(
    name="parallel_analysis",
    sub_agents=[
        gemini_agent.agent,
        grok_agent.agent,
        openai_agent.agent,
    ],
    description="Runs all three MAGI specialists (Gemini, Grok, OpenAI) in parallel "
    "to gather diverse perspectives on the user's query.",
)

# Step 2: SequentialAgent chains parallel analysis -> synthesis
# The synthesizer reads all three outputs and produces a unified response
root_agent = SequentialAgent(
    name="magi_voting_system",
    sub_agents=[
        parallel_analysis,
        synthesizer_agent.agent,
    ],
    description="MAGI voting system that gathers responses from all specialists "
    "and synthesizes them into a unified answer with disagreement analysis.",
)
