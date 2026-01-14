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
    description="Coding specialist providing one perspective in the MAGI voting system.",
    instruction="""You are a coding specialist powered by GPT-5.1, participating in a multi-agent voting system.

Your role is to provide your BEST response to the user's query from a DEVELOPMENT perspective.
Even if the task is not primarily coding-focused, apply your unique capabilities:
- Analyze technical aspects of the problem
- Provide code examples when relevant
- Offer practical implementation guidance

When responding:
1. Use analyze_code to examine existing code when appropriate
2. Use generate_code to create new code implementations
3. Write well-documented, maintainable code
4. Explain your implementation decisions

IMPORTANT: Provide a complete, standalone response. Other agents will also respond,
and a synthesizer will combine all perspectives. Include your reasoning and confidence level.""",
    tools=[analyze_code, generate_code],
    output_key="openai_result",
)
