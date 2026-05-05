"""
评判 prompt：三个维度（Task Fulfillment / Information Grounding / Planning Efficiency）
的 rubric 与 Step 1、Step 2 中的条目可通过 dimension_order 同步打乱顺序。
"""
from __future__ import annotations

from typing import Tuple

# 索引 0=Task Fulfillment, 1=Information Grounding, 2=Planning Efficiency
_RUBRIC_BLOCKS = [
    """## Task Fulfillment
Measures: Did the agent achieve the user's explicit goals?
- 1–3: Perfectly completes 10–30% of requirements.
- 4–6: Perfectly completes 40–60% of requirements.
- 7–8: Perfectly completes 70–80% of requirements.
- 9–10: Perfectly completes 90–100% of requirements.""",
    """## Information Grounding
Measures: Are the agent's final assertions supported by tool outputs?
- 1–3: 10–30% of claims are perfectly grounded in tool outputs.
- 4–6: 40–60% of claims are perfectly grounded in tool outputs.
- 7–8: 70–80% of claims are perfectly grounded in tool outputs.
- 9–10: 90–100% of claims are perfectly grounded in tool outputs.""",
    """## Planning Efficiency
Measures: Did the agent avoid redundant or unnecessary tool calls?
- 9–10: <10% of calls were redundant/unnecessary.
- 7–8: 10–30% of calls were redundant/unnecessary.
- 4–6: 30–60% of calls were redundant/unnecessary.
- 1–3: >60% of calls were redundant/unnecessary.""",
]

_STEP1_LINES = [
    "**Task Fulfillment**: Count the distinct, explicit sub-goals/requirements in the **TASK PRESENTED TO AGENT**.",
    "**Grounding**: Count the atomic factual claims or data points in the agent's FINAL response.",
    "**Planning Efficiency**: Count the TOTAL number of tool calls made in the **AGENT TRAJECTORY FOR TASK COMPLETION**.",
]

_STEP2_BLOCKS = [
    """**Task Fulfillment Issues**: Sub-goals that were FAILED, IGNORED, or INCORRECTLY executed.
   - Note: If a sub-goal was met but via a redundant path, it is STILL COUNTED AS MET for this dimension. Do not penalize efficiency here.""",
    """**Grounding Issues**: Claims in the final answer that are NOT found in or CONTRADICT the tool outputs.""",
    """**Planning Efficiency Issues**: Tool calls that were REDUNDANT (repeating previous successful calls), UNNECESSARY (not needed for any sub-goal), or RETRIED without parameter changes after failure.
   - Note: Necessary calls that were just "slow" or "sub-optimal" but not redundant are NOT issues.""",
]

_JUDGE_PROMPT_PREFIX = """# System Role
You are an impartial evaluator judging the quality of an AI agent's multi-server tool-based task execution. You focus on three independent dimensions: Task Completion, Grounding, and Planning Efficiency.

# User Instructions
You must assign scores based ONLY on evidence from the task, solution, and tool usage trajectory.
- Objective: Ignore language fluency, formatting, or politeness.
- Justified: Every score must be backed by specific counts/percentages derived from the trajectory.
- Independent: Evaluate each dimension separately. A flaw in one dimension (e.g., redundancy) should NOT lower the score of another (e.g., task completion) unless it directly caused failure.

# Input Data
TASK PRESENTED TO AGENT:
{task}

AVAILABLE TOOLS:
{available_tools}

AGENT TRAJECTORY FOR TASK COMPLETION:
{trajectory}

FINAL ANSWER
{answer}

# SCORING RUBRICS (1–10 PER SUBDIMENSION)
Scores are derived from the DEFECT RATE calculated in the principles below.

"""

_JUDGE_PROMPT_MID = """

# CALCULATION METHODOLOGY

## Step 1: Define "Opportunities" (Denominator) for Each Dimension
"""

_JUDGE_PROMPT_STEP2_HEADER = """

## Step 2: Identify "Issues" (Numerator) for Each Dimension
"""

_JUDGE_PROMPT_SUFFIX = """

## Step 3: Calculate Defect Rate & Map to Score
- Defect Rate = (Issues / Opportunities) × 100%
- Mapping:
  - 0–10% defects → Score 9–10
  - 10–30% defects → Score 7–8
  - 30–60% defects → Score 4–6
  - 60–100% defects → Score 1–3

# How to Score:
1. When evaluating percentages, be EXTREMELY STRICT about what counts as "perfectly executed".
2. "Perfectly" means ALL of the following must be true:
    - Complete and accurate parameters (not just valid, but IDEAL)
    - Zero redundancy (no repeated or unnecessary calls)
    - Proper error handling (graceful recovery from ANY failure)
    - Efficient execution (minimal rounds)
3. If ANY of the above is missing, that portion is NOT perfectly executed (counts as 0%).
4. Example: Task completed correctly but with 1 redundant call = that portion is 0% perfect.

# KEY PRINCIPLES:
1. ALWAYS calculate as percentage, NOT absolute numbers.
2. 10 errors in 100 calls (10%) = same score as 1 error in 10 calls (10%).
3. NORMALIZE by complexity - don’t punish complex tasks:
    - Simple task: 1 error/5 steps (20% defect) = Score 7
    - Complex task: 4 errors/20 steps (20% defect) = Score 7
4. CRITICAL: Apply the STRICTEST interpretation of “perfectly executed”. If there’s ANY doubt, score lower.

# FINAL REMINDER BEFORE SCORING:
- Count ONLY truly perfect executions toward the percentage
- Be your most critical self - find flaws first, then acknowledge successes
- If you're considering a score above 8, re-examine for ANY imperfection
- Server count is IRRELEVANT - using more servers is NOT better

# OUTPUT FORMAT
Return your evaluation scoring and reasoning in this exact JSON format. Return ONLY the JSON object.
```json
{{
  "task_fulfillment_reasoning": "List total requirements vs. met requirements. Calculate defect rate.",
  "grounding_reasoning": "List total claims vs. unsupported claims. Calculate defect rate.",
  "planning_efficiency_reasoning": "List total tool calls vs. redundant/unnecessary calls. Calculate defect rate.",
  "task_fulfillment": <integer_score>,
  "grounding": <integer_score>,
  "planning_and_efficiency": <integer_score>
}}
```
"""


def _numbered_lines(lines: list[str], order: Tuple[int, int, int]) -> str:
    return "\n".join(f"{j + 1}. {lines[idx]}" for j, idx in enumerate(order))


def _numbered_step2_blocks(order: Tuple[int, int, int]) -> str:
    return "\n".join(f"{j + 1}. {_STEP2_BLOCKS[idx]}" for j, idx in enumerate(order))


def build_mcp_bench_judge_prompt(
    task: str,
    available_tools: str,
    trajectory: str,
    answer: str,
    *,
    dimension_order: Tuple[int, int, int] = (0, 1, 2),
) -> str:
    """
    按 dimension_order 同步打乱三个 rubric 块以及 Step 1 / Step 2 中三条描述的顺序。
    元组为三个下标 0,1,2 的一个排列，表示「先写哪一维、再写哪一维」。
    """
    o = dimension_order
    rubrics = "\n\n".join(_RUBRIC_BLOCKS[i] for i in o)
    step1_body = _numbered_lines(_STEP1_LINES, o)
    step2_body = _numbered_step2_blocks(o)
    return (
        _JUDGE_PROMPT_PREFIX
        + rubrics
        + _JUDGE_PROMPT_MID
        + step1_body
        + _JUDGE_PROMPT_STEP2_HEADER
        + step2_body
        + _JUDGE_PROMPT_SUFFIX.format()
    ).replace("{task}", task).replace("{available_tools}", available_tools).replace(
        "{trajectory}", trajectory
    ).replace("{answer}", answer)


# 默认顺序的完整模板（占位符未替换），兼容旧用法：仍用 .replace 填字段
MCP_BENCH_JUDGE_PROMPT = build_mcp_bench_judge_prompt(
    "{task}",
    "{available_tools}",
    "{trajectory}",
    "{answer}",
    dimension_order=(0, 1, 2),
)
