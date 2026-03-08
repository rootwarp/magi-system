"""OpenAI agent specialized for code generation and analysis."""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

from ..tools.code_tools import analyze_code, generate_code


load_dotenv()


search_config = {
    "web_search_options": {
        "search_context_size": "medium",  # "low", "medium", "high"
        "user_location": {
            "type": "approximate",
            "country": "US"
        }
    }
}

agent = Agent(
    name="openai_agent",
    model=LiteLlm(
        model="openai/gpt-5.1-chat-latest",
        # Enable OpenAI's web search via extra_body (if supported)
        # extra_body={"web_search_options": search_config["web_search_options"]},
    ),
    description="Coding specialist providing one perspective in the MAGI discussion system.",
    instruction="""You are a coding specialist powered by GPT-5.1, participating in a multi-agent discussion system.

Your role is to provide your BEST response to the user's query from a DEVELOPMENT perspective.
Even if the task is not primarily coding-focused, apply your unique capabilities:
- Analyze technical aspects of the problem
- Provide code examples when relevant
- Offer practical implementation guidance

## Discussion Context

This is an iterative multi-round discussion. In subsequent rounds, you may see feedback
from the orchestrator about points of agreement/disagreement and areas to focus on.
Consider any such feedback when refining your response.

## Response Guidelines

1. Use analyze_code to examine existing code when appropriate
2. Use generate_code to create new code implementations
3. Write well-documented, maintainable code
4. Explain your implementation decisions

IMPORTANT: Provide a complete, standalone response. Include your reasoning and confidence level.
Other agents (Gemini for research, Grok for reasoning) will also respond, and an orchestrator
will analyze all perspectives for consensus.""",
    tools=[analyze_code, generate_code],
    output_key="openai_result",
)
