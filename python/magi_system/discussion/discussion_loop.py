"""Discussion loop: iterative multi-agent discussion with termination criteria."""

from google.adk.agents import LoopAgent
from dotenv import load_dotenv

from .discussion_round import discussion_round

load_dotenv()

# Maximum number of discussion rounds before forced termination
MAX_ITERATIONS = 3

loop_agent = LoopAgent(
    name="discussion_loop",
    max_iterations=MAX_ITERATIONS,
    sub_agents=[discussion_round],
    description=f"Iterative discussion loop that runs up to {MAX_ITERATIONS} rounds. "
    "Terminates early when orchestrator detects consensus or calls exit_loop().",
)
