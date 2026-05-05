TASK_QUALITY_ASSESSMENT_PROMPT = """
Purpose: Evaluate task quality on solvability and utility dimensions

Input Data:
User Query:
{user_query}

Available MCP Servers & Tools:
{tool_descriptions}

EVALUATION CRITERIA:

1. SOLVABILITY (1-10):
Assess whether the agent can successfully complete the task using ONLY the provided tools and the information in the query.
• 10: All required data is provided, tools perfectly match needs, clear success criteria. No external dependencies needed.
• 8-9: Task is clearly solvable with the given tools. Minor ambiguities exist but are resolvable by reasonable agent reasoning.
• 6-7: Mostly solvable but some steps may be challenging, unclear, or require iterative attempts due to vague tool outputs.
• 4-5: Significant gaps in tool coverage or data requirements. Key tools are missing or data is inaccessible.
• 1-3: Task cannot be meaningfully completed with available tools. Requires external resources, physical actions, or private data not exposed.

Consider:
• Are multiple MCP servers required to coordinate in order to complete the task?
• Can the agent achieve the stated goal based on the function and output of these given tools?
• Are success criteria clear and measurable?

2. PRACTICAL UTILITY (1-10):
Assess whether the query addresses a genuine user need rather than being a contrived, artificial, or trivial test case.
• 10: Critical business/research value. Addresses a real-world problem perfectly. The outcome is highly actionable and insightful.
• 8-9: Strong practical value. Useful for decision-making, operations, or learning. Reflects a common or important scenario.
• 6-7: Moderate value. Interesting or educational, but perhaps niche. Not immediately critical but still meaningful.
• 4-5: Limited practical value. Feels like a synthetic benchmark task or academic exercise with little real-world application.
• 1-3: Trivial or artificial task. No real-world application (e.g., simple arithmetic, listing tools). Contrived scenarios no human would ask.

Consider:
• Does this address a real business, research, or personal productivity need?
• Would the results be actionable and valuable to a professional or general user?
• Does it avoid "toy problems" that only exist to test syntax rather than reasoning?
• Does it sound like a natural human request rather than a robotic instruction?

OUTPUT FORMAT:
Provide scores and brief feedback in JSON format ONLY:
```json
{{
  "solvability_score": <number 1-10>,
  "utility_score": <number 1-10>,
  "solvability_feedback": "<Brief explanation focusing on tool coverage, data availability, and logical flow>",
  "utility_feedback": "<Brief explanation focusing on realism, business/research value, and naturalness>"
}}
```
"""