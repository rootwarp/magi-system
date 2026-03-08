"""Search fan-out module for the MAGI system.

Handles decomposing research plans into sub-questions and dispatching
them to multiple search agents in parallel.
"""

from __future__ import annotations

import re
from typing import Union

from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import google_search

from ..config import ModelConfig
from ..tools.content_extraction import extract_page_content

SEARCH_INSTRUCTION_TEMPLATE = """\
You are a search agent researching a specific sub-question as part of a \
larger research effort.

RESEARCH BRIEF:
{research_brief}

YOUR ASSIGNED SUB-QUESTION:
{sub_question}

INSTRUCTIONS:
1. Search thoroughly for information relevant to your sub-question.
2. When you find relevant URLs, use extract_page_content to read the full \
content.
3. Evaluate sources for credibility and recency.
4. Produce evidence-backed claims in the structured format below.

For each finding, output a claim in this exact format:

CLAIM: [factual claim]
SOURCE: [URL]
CONFIDENCE: [high/medium/low]
EVIDENCE: [supporting quote or summary]

Produce as many well-supported claims as you can find. Focus on accuracy \
and source quality over quantity.\
"""


def _get_tools_for_model(model_cfg: ModelConfig) -> list:
    """Return the appropriate tools list for a given model configuration.

    Gemini models get google_search and extract_page_content.
    Grok models rely on native search (via extra_body) so only get
    extract_page_content. GPT and Claude models also only get
    extract_page_content.

    Args:
        model_cfg: The model configuration to select tools for.

    Returns:
        List of tool functions for the model's search agent.
    """
    if "gemini" in model_cfg.name.lower():
        return [google_search, extract_page_content]
    return [extract_page_content]


def _get_model_for_search(
    model_cfg: ModelConfig,
) -> Union[str, LiteLlm]:
    """Return the model instance configured for search use.

    Grok models are wrapped in a new LiteLlm instance with
    search_parameters in extra_body to enable native search.
    All other models are returned as-is.

    Args:
        model_cfg: The model configuration to prepare for search.

    Returns:
        A string model ID or LiteLlm instance ready for search agent use.
    """
    if "grok" in model_cfg.name.lower():
        model_id = (
            model_cfg.model.model
            if isinstance(model_cfg.model, LiteLlm)
            else model_cfg.model
        )
        return LiteLlm(
            model=model_id,
            extra_body={"search_parameters": {"mode": "auto"}},
        )
    return model_cfg.model


def _parse_sub_questions(plan: str) -> list[str]:
    """Extract sub-questions from a planner's text-based output.

    Parses lines matching SQ{N} patterns (e.g., "SQ1: ...", "SQ 1: ...",
    "SQ1 - ...") and returns the question text. Metadata lines like
    PRIORITY and DEPENDS_ON are ignored.

    Args:
        plan: Raw text output from the planner containing SQ lines.

    Returns:
        List of extracted question strings. Empty list if none found.
    """
    pattern = re.compile(r"^\s*SQ\s*\d+\s*[:\-]\s*(.+?)\s*$")
    questions: list[str] = []
    for line in plan.splitlines():
        match = pattern.match(line)
        if match:
            question = match.group(1).strip()
            if question:
                questions.append(question)
    return questions
