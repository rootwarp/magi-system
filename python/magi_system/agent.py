"""MAGI System research pipeline using ADK Workflow Agents.

This is the main entry point for the MAGI multi-agent system.
The pipeline processes queries through three stages:
1. Clarifier refines the user query into a structured research brief
2. Planner decomposes the brief into prioritized sub-questions
3. DynamicSearchFanout dispatches parallel search agents per sub-question
"""

from google.adk.agents import SequentialAgent
from dotenv import load_dotenv

from .clarification import clarifier_agent
from .config import load_config
from .planning import planner_agent
from .search import DynamicSearchFanout

load_dotenv()

config = load_config()

search_fanout = DynamicSearchFanout(
    name="search_fanout",
    config=config,
)

# MAGI Research Pipeline:
# Step 1: Clarifier refines user query into a research brief
# Step 2: Planner decomposes brief into sub-questions
# Step 3: DynamicSearchFanout fans out search across models per sub-question
root_agent = SequentialAgent(
    name="magi_research_pipeline",
    sub_agents=[
        clarifier_agent.agent,
        planner_agent,
        search_fanout,
    ],
    description="MAGI research pipeline that clarifies queries, plans sub-questions, "
    "and fans out parallel search across multiple models.",
)

# --- Previous pipeline (MAGI Discussion System) ---
# Preserved for rollback reference.
#
# from .discussion import loop_agent
# from .sub_agents import synthesizer_agent
#
# root_agent = SequentialAgent(
#     name="magi_discussion_system",
#     sub_agents=[
#         loop_agent,
#         synthesizer_agent.agent,
#     ],
#     description="MAGI discussion system that runs iterative multi-agent discussions "
#     "with orchestrated consensus detection, then synthesizes final response.",
# )
