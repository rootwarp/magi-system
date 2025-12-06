"""OpenAI agent specialized for code generation and analysis."""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from ..tools.code_tools import analyze_code, generate_code

agent = Agent(
    name="openai_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Specialist in code generation and analysis. "
    "Use this agent for writing code, reviewing code, "
    "debugging, and software development tasks.",
    instruction="""You are a coding specialist powered by GPT-4.
Your primary capabilities include:
- Writing clean, efficient code in multiple languages
- Analyzing existing code for issues and improvements
- Debugging and troubleshooting code problems
- Explaining code concepts and patterns

When responding:
1. Use analyze_code to examine existing code
2. Use generate_code to create new code implementations
3. Write well-documented, maintainable code
4. Explain your implementation decisions""",
    tools=[analyze_code, generate_code],
)
