"""Discussion loop with AgentTool for resolving divergences.

MAGI-022: Implements a moderator-led discussion loop where AgentTool-wrapped
discussion agents debate divergence points, followed by a resolution evaluator
that decides whether to continue or exit the loop.
"""

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.exit_loop_tool import exit_loop

from ..config import PipelineConfig

ROUTER_INSTRUCTION = """You are a discussion moderator resolving research divergences.

Divergence Map:
{divergence_map}

For each NEEDS_DISCUSSION item in the divergence map:
1. Present the divergence to relevant discussion participants
2. Ask each participant to provide evidence-backed positions
3. Facilitate constructive debate
4. Record each participant's final position

Use the available agent tools to consult with discussion participants.
Summarize all positions and evidence gathered."""

EVALUATOR_INSTRUCTION = """You are a resolution evaluator assessing discussion outcomes.

Divergence Map:
{divergence_map}

Discussion Results:
{discussion_round_result}

For each divergence that was discussed:
- If participants converged on a position with evidence: mark as RESOLVED
- If participants remain divided despite discussion: mark as DISPUTED

Output format:
=== RESOLUTION STATUS ===
[For each divergence item]
ITEM: [description]
STATUS: RESOLVED | DISPUTED
EVIDENCE: [key evidence cited]
FINAL_POSITION: [consensus position or both positions if disputed]

If ALL items are either RESOLVED or DISPUTED, call exit_loop() to end the discussion.
If some items might benefit from another round, do NOT call exit_loop()."""


def create_discussion_loop(config: PipelineConfig) -> LoopAgent:
    """Create a discussion loop for resolving divergences.

    Creates fresh agent instances each time (ADK single-parent rule:
    agents can only belong to one parent agent graph).

    Args:
        config: Pipeline configuration with model settings.

    Returns:
        LoopAgent wrapping discussion router and resolution evaluator.
    """
    discussion_agents = []
    for model_cfg in config.search_models:
        if not model_cfg.enabled:
            continue

        model = (
            model_cfg.model
            if "gemini" in model_cfg.name.lower()
            else LiteLlm(model=model_cfg.model)
        )
        agent = LlmAgent(
            name=f"discussant_{model_cfg.name}",
            model=model,
            instruction="""You are a research discussion participant.
When asked about a divergence point, provide your evidence-backed position.
Cite specific sources and explain your reasoning.
If presented with compelling counter-evidence, update your position.""",
            description=f"Discussion participant using {model_cfg.name}.",
        )
        discussion_agents.append(agent)

    agent_tools = [AgentTool(agent=agent) for agent in discussion_agents]

    discussion_router = LlmAgent(
        name="discussion_router",
        model="gemini-3-pro-preview",
        instruction=ROUTER_INSTRUCTION,
        tools=agent_tools,
        output_key="discussion_round_result",
        description="Moderates discussion rounds between model agents.",
    )

    resolution_evaluator = LlmAgent(
        name="resolution_evaluator",
        model="gemini-3-pro-preview",
        instruction=EVALUATOR_INSTRUCTION,
        tools=[exit_loop],
        output_key="consensus_result",
        description="Evaluates if discussion has converged and decides to exit loop.",
    )

    discussion_round = SequentialAgent(
        name="discussion_round",
        sub_agents=[discussion_router, resolution_evaluator],
        description="One round of discussion followed by resolution evaluation.",
    )

    return LoopAgent(
        name="discussion_loop",
        sub_agents=[discussion_round],
        max_iterations=config.max_discussion_rounds,
        description="Iterative discussion loop for resolving divergences.",
    )
