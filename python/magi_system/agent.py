"""MAGI System research pipeline using ADK Workflow Agents.

Full 6-step pipeline:
1. Clarifier refines the user query into a structured research brief
2. Planner decomposes the brief into prioritized sub-questions
3-5. Research loop (search fanout → aggregator → reflection → consensus)
6. Synthesizer produces the final structured research report
"""

from google.adk.agents import SequentialAgent
from dotenv import load_dotenv

from .clarification import clarifier_agent
from .config import load_config
from .pipeline import create_research_loop
from .planning import planner_agent
from .synthesis import synthesizer_agent

load_dotenv()

config = load_config()

research_loop = create_research_loop(config)

root_agent = SequentialAgent(
    name="magi_research_pipeline",
    sub_agents=[
        clarifier_agent.agent,
        planner_agent,
        research_loop,
        synthesizer_agent,
    ],
    description="MAGI research pipeline: clarify → plan → research loop → synthesize.",
)
