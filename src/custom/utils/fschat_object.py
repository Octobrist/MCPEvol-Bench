import json
import re
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- 1. 定义模拟 OpenAI SDK 的数据结构 ---

class FunctionCall(BaseModel):
    """模拟 openai.types.chat.ChatCompletionMessageToolCall.Function"""
    name: str
    arguments: str  # 注意：OpenAI 标准中 arguments 是 JSON 字符串


class ToolCall(BaseModel):
    """模拟 openai.types.chat.ChatCompletionMessageToolCall"""
    id: str
    type: str = "function"
    function: FunctionCall


class AssistantMessage(BaseModel):
    """
    模拟 openai.types.chat.ChatCompletionMessage
    用于承载 LLM 的返回结果，兼容你主函数中的属性访问
    """
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    def model_dump(self, exclude_none: bool = True) -> Dict[str, Any]:
        """
        兼容 Pydantic v2 的 model_dump 行为，转换为字典以便存入 history
        """
        # 手动构建字典以确保格式完全符合 OpenAI API 标准
        data = {"role": self.role}

        if self.content is not None:
            data["content"] = self.content

        if self.tool_calls:
            # 将 ToolCall 对象列表转换为字典列表
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in self.tool_calls
            ]

        if exclude_none:
            return {k: v for k, v in data.items() if v is not None}
        return data