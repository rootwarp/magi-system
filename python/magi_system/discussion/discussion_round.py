"""Discussion round: one iteration of parallel specialist analysis + orchestrator evaluation."""

from google.adk.agents import ParallelAgent, SequentialAgent
from dotenv import load_dotenv

from ..sub_agents import gemini_agent, grok_agent, openai_agent, orchestrator_agent

load_dotenv()

# Step 1: Run all specialists in parallel
parallel_specialists = ParallelAgent(
    name="parallel_specialists",
    sub_agents=[
        gemini_agent.agent,
        grok_agent.agent,
        # openai_agent.agent,  # Enable when needed
    ],
    description="Runs all MAGI specialists in parallel to gather diverse perspectives.",
)

# Step 2: Orchestrator evaluates responses and decides continue/terminate
discussion_round = SequentialAgent(
    name="discussion_round",
    sub_agents=[
        parallel_specialists,
        orchestrator_agent.agent,
    ],
    description="One round of discussion: parallel specialist responses followed by orchestrator evaluation.",
)
