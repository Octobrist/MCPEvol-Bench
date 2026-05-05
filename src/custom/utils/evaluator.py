import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI, BadRequestError, APIError

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def truncate_tool_content(messages: List[Dict[str, Any]], max_length: int = 1000) -> List[Dict[str, Any]]:
    """
    遍历消息列表，当角色的 role 为 'tool' 时，截断其 content 内容。

    参数:
        messages: 包含字典的列表，每个字典代表一条消息 (通常包含 'role' 和 'content' 键)
        max_length: 保留的最大字符数，默认 1000

    返回:
        处理后的新列表 (注意：为了安全起见，这里返回的是新列表，不修改原数据)
    """
    processed_messages = []
    for msg in messages:
        # 创建副本以避免修改原始数据
        new_msg = msg.copy()
        # 检查是否为 tool 角色且存在 content
        if new_msg.get("role") == "tool" and isinstance(new_msg.get("content"), str):
            content = new_msg["content"]
            # 如果内容超过限制，进行截断
            if len(content) > max_length:
                new_msg["content"] = content[:max_length] + "..."
        elif new_msg.get("role") == "assistant" and isinstance(new_msg.get("content"), str):
            new_msg['content'] = ""
        processed_messages.append(new_msg)
    return processed_messages

class AgentEvaluator:
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the LLM-as-a-Judge evaluator for agent task completion.

        :param api_key: OpenAI API Key (falls back to OPENAI_API_KEY env var)
        :param model: Model name for evaluation (default: gpt-4o)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        if not self.api_key:
            raise ValueError("OpenAI API Key is required. Set it via argument or OPENAI_API_KEY env var.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.model = model
# '3. **Hallucination Check**: Does the final output contain claims that are not supported by the tool responses or context, without explicit disclaimer?'
    def _build_judge_prompt(self, instruction: str, trajectory: List[Dict], final_output: str) -> List[Dict]:
        """
        Build the prompt for the Judge LLM.
        """
        system_prompt = """
You are an expert evaluator for AI agent task completion (LLM-as-a-Judge).
Your task is to determine whether the agent has successfully fulfilled the user's request.

Evaluation Criteria:
1. **Intent Alignment**: Does the final output directly address the user's core question or complete the core task?
2. **Tool Usage**: Did the agent correctly invoke the necessary tools? Is the final conclusion supported by the tool responses?
3. **Completeness**: Are any key constraints from the user instruction (e.g., time, location, format) overlooked?
4. **Correctness**: If a tool call returns an uninformative result, such as content=[TextContent(type='text', text='[]'...)] or "[TextContent(type='text', text="Error" ...)]", it can be considered as providing no support for the answer.

Note that the tool output has been truncated, and some content may have been omitted by "...".

Output Format:
Respond with a STRICT JSON object only. Do NOT include markdown, explanations, or any other text.
{
    "is_success": boolean,  // true if the task is successfully completed, false otherwise
    "reason": string        // concise justification for the judgment
}
"""

        # Serialize trajectory for LLM consumption
        trajectory_text = json.dumps(truncate_tool_content(trajectory), ensure_ascii=False, indent=2)

        user_prompt = f"""### User Instruction
{instruction}

### Interaction Trajectory
{trajectory_text}

### Agent Final Output
{final_output}

Please evaluate based on the criteria above and output the JSON result.
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def evaluate(self, instruction: str, trajectory: List[Dict], final_output: str) -> Dict[str, Any]:
        """
        Execute the evaluation.

        :param instruction: The original user instruction
        :param trajectory: List of messages in OpenAI format (including tool_calls and tool responses)
        :param final_output: The agent's final response to the user
        :return: Dict with 'is_success' (bool) and 'reason' (str)
        """
        messages = self._build_judge_prompt(instruction, trajectory, final_output)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,  # Deterministic evaluation
                max_tokens=2048
            )
            content = response.choices[0].message.content
            match = re.search(r'```(?:json)?\s*({.*?})\s*```', content, re.DOTALL | re.IGNORECASE)
            if match:
                result = json.loads(match.group(1))
            else:
                result = json.loads(content)
            # Validate required fields
            if "is_success" not in result:
                logger.warning("Judge LLM missing 'is_success' field. Defaulting to False.")
                result["is_success"] = False
            return result

        except BadRequestError as e:
            # 专门处理 400 错误 (包括上下文超长)
            error_code = None
            error_msg = str(e)

            # 尝试从 e.body 中提取详细的错误代码
            if hasattr(e, 'body') and isinstance(e.body, dict):
                error_info = e.body.get('error', {})
                error_code = error_info.get('code')
                if 'message' in error_info:
                    error_msg = error_info['message']

            if error_code == 'context_length_exceeded':
                logger.warning(f"Evaluation skipped: Context length exceeded for instruction '{instruction[:50]}...'")
                return {
                    "is_success": False,
                    "reason": f"context_length_exceeded: {error_msg}"
                }

            # 其他 400 错误 (如参数格式错误)
            logger.error(f"Evaluation failed (Bad Request): {error_msg}")
            return {
                "is_success": False,
                "reason": f"invalid_request_error: {error_msg}"
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Judge response as JSON: {e}")
            return {
                "is_success": False,
                "reason": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "is_success": False,
                "reason": f"Evaluation system error: {str(e)}"
            }