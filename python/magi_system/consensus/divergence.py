"""Divergence detection agent for consensus analysis.

MAGI-021: Analyzes search results and model reflections to identify
areas of agreement and divergence, with a voting fast path for
resolving 2/3 majority agreements.
"""

from google.adk.agents import LlmAgent

DIVERGENCE_INSTRUCTION = """You are a divergence analysis specialist. Your job is to compare \
search results and model reflections to identify where they agree and where they diverge.

## Input

Research Plan:
{research_plan}

Gemini Reflection:
{reflection_gemini}

Claude Reflection:
{reflection_claude}

Search Results:
{all_search_results}

## Analysis Process

1. **Identify Agreement Zones**: Find claims where 2 or more models/sources agree. \
Assign a confidence score (0.0-1.0) based on the strength and consistency of agreement.

2. **Detect Divergences**: For each point of disagreement, classify the divergence type:
   - CONTRADICTION: Models make directly opposing claims
   - MAGNITUDE: Models agree on direction but disagree on scale or degree
   - OMISSION: One model covers a topic that another ignores entirely
   - RANKING: Models agree on factors but rank their importance differently

3. **Apply Voting Fast Path**: When 2/3 majority of sources agree on a claim, \
mark it as RESOLVED_BY_VOTE. This avoids unnecessary discussion on points where \
there is clear consensus despite one dissenting source.

4. **Flag Remaining Divergences**: Items not resolved by vote are flagged as \
NEEDS_DISCUSSION with a severity level:
   - HIGH: Core factual contradiction or critical disagreement affecting conclusions
   - MEDIUM: Meaningful difference in interpretation or emphasis
   - LOW: Minor nuance or peripheral disagreement

5. **Threshold-Based Skip**: If the overall agreement score exceeds the threshold \
and there are no HIGH-severity divergences, set DISCUSSION_NEEDED to no. This \
allows the pipeline to skip the discussion phase when consensus is already strong.

## Output Format

=== AGREEMENT ZONE ===
[List each claim where models agree]
- Claim: [description]
  Confidence: [0.0-1.0]
  Sources: [which models/sources agree]

=== RESOLVED BY VOTE ===
[Claims where 2/3 majority resolves the divergence]
- Claim: [description]
  Majority position: [the agreed-upon position]
  Dissenting source: [which source disagrees]

=== NEEDS DISCUSSION ===
[Remaining divergences requiring deliberation]
- Divergence: [description]
  Type: CONTRADICTION | MAGNITUDE | OMISSION | RANKING
  Severity: HIGH | MEDIUM | LOW
  Positions: [summary of each model's position]

=== SUMMARY ===
OVERALL_AGREEMENT_SCORE: [0.0-1.0]
DISCUSSION_NEEDED: [yes|no]
"""

divergence_detector = LlmAgent(
    name="divergence_detector",
    model="gemini-3-pro-preview",
    instruction=DIVERGENCE_INSTRUCTION,
    output_key="divergence_map",
    description="Detects agreement and divergence zones across model outputs "
    "with voting fast path for 2/3 majority resolution.",
)
