"""MAGI System discussion architecture using ADK Workflow Agents.

This is the main entry point for the MAGI multi-agent system.
All requests are processed through an iterative discussion loop where:
1. Multiple specialists respond in parallel
2. An orchestrator analyzes responses for consensus
3. Discussion continues until consensus or max iterations
4. Final synthesizer produces unified response
"""

from google.adk.agents import SequentialAgent
from dotenv import load_dotenv

from .discussion import loop_agent
from .sub_agents import synthesizer_agent


load_dotenv()

# MAGI Discussion System:
# Step 1: LoopAgent runs iterative discussion rounds
#   - Each round: ParallelAgent (specialists) -> Orchestrator (evaluation)
#   - Terminates when orchestrator calls exit_loop() or max_iterations reached
# Step 2: Synthesizer produces final unified response from discussion
root_agent = SequentialAgent(
    name="magi_discussion_system",
    sub_agents=[
        loop_agent,
        synthesizer_agent.agent,
    ],
    description="MAGI discussion system that runs iterative multi-agent discussions "
    "with orchestrated consensus detection, then synthesizes final response.",
)
