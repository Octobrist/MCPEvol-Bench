import asyncio
import json
import uuid

from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageToolCall

from custom.utils.fastchat_utils import chat_with_fastchat
from custom.utils.test_utils import offset_date, generate_tree_string

SINGLE_SERVER_INTERATION_SYSTEM_PROMPT = '''You are an AI agent connected to the MCP (Model Context Protocol) server. Based on the user's request, select and invoke tools to fulfill the task accurately and efficiently. 
Note that do not to ask any questions to the user, just call the tool to complete the task.

Considering real-time performance, the time is set to be around `{SET_TIME}`.

These files are stored in the local file system and serve as relevant resources for tool utilization. Add the prefix path "./anotation_path" to the following file systems as an absolute path during use:
`{FILE_EXISTS}`

Database Configuration (if applicable):
- Host: 127.0.0.1
- Port: 3305
- User: root
- Password: 123456
- Database Name: mydb
'''

from typing import List, cast
import logging
from openai import OpenAI
from functools import partial
from backoff import on_exception, expo
import os

logger = logging.getLogger(__name__)

import mcp
from mcp.client.streamable_http import streamablehttp_client

import asyncio
import base64
import json
import os
from typing import Callable, Any


def convert_to_claude_messages(openai_messages):
    """
    将 OpenAI 风格的消息列表 (包含 role='tool')
    转换为 Anthropic Claude 风格 (role='user' + tool_result 块)
    """
    claude_messages = []
    for msg in openai_messages:
        m_dict = msg if isinstance(msg, dict) else msg.model_dump(exclude_none=True)
        role = m_dict.get("role")
        content = m_dict.get("content")
        if role == "assistant":
            new_content = []
            if content:
                new_content.append({"type": "text", "text": content})
            tool_calls = m_dict.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    tc_dict = tc if isinstance(tc, dict) else tc.model_dump(exclude_none=True)
                    if not tc_dict.get("id"):
                        raise ValueError("Critical: Assistant message has tool_use without ID!")
                    new_content.append({
                        "type": "tool_use",
                        "id": tc_dict["id"],
                        "name": tc_dict["function"]["name"],  # OpenAI 结构
                        "input": json.loads(tc_dict["function"]["arguments"]) if isinstance(
                            tc_dict["function"]["arguments"], str) else tc_dict["function"]["arguments"]
                    })
            claude_messages.append({"role": "assistant", "content": new_content})
        elif role == "tool":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": m_dict["tool_call_id"],  # 映射字段名
                "content": m_dict["content"]
            }
            if claude_messages and claude_messages[-1]["role"] == "user":
                last_content = claude_messages[-1]["content"]
                if isinstance(last_content, list):
                    last_content.append(tool_result_block)
                else:
                    claude_messages[-1]["content"] = [{"type": "text", "text": last_content}, tool_result_block]
            else:
                claude_messages.append({
                    "role": "user",
                    "content": [tool_result_block]
                })

        elif role == "user":
            if isinstance(content, list):
                claude_messages.append({"role": "user", "content": content})
            else:
                claude_messages.append({"role": "user", "content": [{"type": "text", "text": content}]})
    return claude_messages


# --- 配置 ---
# MCP_SERVER_URL = "http://127.0.0.1:3050/mcp"  # 根据日志确定端口为 3050
TOKEN = None  # 日志显示 "Authentication disabled"，故无需 token

async def call_tool_with_1mcp(name, arguments, url):
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with mcp.ClientSession(read_stream, write_stream) as session:
            # 初始化会话
            await session.initialize()
            result = await session.call_tool(
                name=name,
                arguments=arguments
            )
    return result

class ChatModel:
    def __init__(
        self,
        model_name=None,
        model_url=None,
        api_key=None,
        temperature=0.2,
        max_new_tokens=4096,
        tools=[],
        mcp_client=None,
        max_round=15,
        tool_server_map=None,
        file_exists='',
        mcp_server_url='http://127.0.0.1:3050/mcp'
    ):
        self.model_name = model_name
        self.model_url = model_url
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url=model_url,
        )
        self.url = mcp_server_url
        self.extra_body = {"enable_thinking": True}
        # self.init_extra_body()
        self.chat = partial(
            self.client.chat.completions.create,
            model=model_name,
            temperature=temperature,
            max_completion_tokens=max_new_tokens,
            # extra_body=self.extra_body,
        )
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": tool['inputSchema'],
                },
            }
            for tool in tools
        ]

        self.mcp_client = mcp_client
        self.messages = [{'role':'system', 'content':SINGLE_SERVER_INTERATION_SYSTEM_PROMPT.format(
            SET_TIME=str(offset_date(0)),
            FILE_EXISTS=file_exists
        )
}]
        self.max_round = max_round
        self.tool_server_map = tool_server_map

    def init_extra_body(self):
        self.extra_body["enable_thinking"] = False

    def chat_with_retry(self, message, retry=4):
        @on_exception(expo, Exception, max_tries=retry)
        def _chat_with_retry(message):
            return self.chat(messages=message)

        try:
            response = _chat_with_retry(message)
            return response
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise e

    def complete_with_retry(self, **args):
        @on_exception(expo, Exception, max_tries=5)
        def _chat_with_retry(**args):
            return self.chat(**args)

        try:
            response = _chat_with_retry(**args)
            return response
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise e

    def list_models(self):
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise e

    def run_with_server(self, query, fschat_flag=False):
        self.messages.append({'role':'user', 'content':query})
        no_tool_count = 0
        final_text = []
        stop_flag = False
        exceeded_flag = False
        round_num = 0
        try:
            while not stop_flag and round_num <= self.max_round:
                raw_messages = self.messages
                if not fschat_flag:
                    request_payload = {
                        "messages": raw_messages,
                        "tools": self.tools,
                    }
                    response = self.complete_with_retry(**request_payload)
                    if hasattr(response, "error"):
                        raise Exception(
                            f"Error in OpenAI response: {response.error['metadata']['raw']}"
                        )
                    response_message = response.choices[0].message
                else:
                    response_message = chat_with_fastchat(this_messages=self.messages, model_name=self.model_name, tools=self.tools)

                round_num += 1
                if response_message.tool_calls:
                    tool_call_list = []
                    for tool_call in response_message.tool_calls:
                        if not tool_call.id:
                            tool_call.id = str(uuid.uuid4())
                        tool_call_list.append(tool_call)
                    response_message.tool_calls = tool_call_list

                self.messages.append(response_message.model_dump(exclude_none=True))
                if hasattr(response_message, 'content'):
                    content = response_message.content
                else:
                    content = str(response_message)

                if content is not None and '[FINISH]' in content:
                    final_text.append(content)
                    stop_flag = True
                elif (
                        content
                        and not response_message.tool_calls
                        # and not response_message.function_call
                ):
                    no_tool_count += 1
                    if no_tool_count == 3:
                        final_text.append(content)
                        stop_flag = True
                    else:
                        self.messages.append(
                            {
                                "role": "user",
                                "content": 'Please call the tool directly to complete the task without asking me any questions.',
                            }
                        )
                else:
                    tool_calls = response_message.tool_calls
                    if not tool_calls:
                        logger.warning(
                            "Received empty response from LLM without content or tool calls."
                        )
                        break
                    no_tool_count = 0
                    for tool_call in tool_calls:
                        try:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)
                            tool_id = tool_call.id
                            tool_server_name = self.tool_server_map[tool_name]
                            tool_1mcp_name = f'{tool_server_name}_1mcp_{tool_name}'
                            # if 'music-analysis' in tool_1mcp_name:
                            #     result = asyncio.run(call_tool_with_1mcp(tool_1mcp_name, tool_args, 'http://127.0.0.1:13050:mcp'))
                            # else:
                            result = asyncio.run(call_tool_with_1mcp(tool_1mcp_name, tool_args, self.url))


                        except asyncio.TimeoutError:
                            logger.error(f"Tool call {tool_name} timed out.")
                            result = "Tool call timed out."
                            tool_id = None
                        except Exception as e:
                            logger.error(f"Error calling tool {tool_name}: {e}")
                            result = f"Error: {str(e)}"
                            tool_id = tool_call.id
                        if hasattr(result, 'content'):
                            result = str(result.content)[:500]+'...'
                        else:
                            result = str(result)[:500]+'...'
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": result,
                            }
                        )
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            final_text.append(f"Error: {str(e)}")
            self.messages.append({"role": "assistant", "content": str(e)})
        if round_num >= self.max_round:
            exceeded_flag = True
        return "\n".join(final_text), self.messages, stop_flag, exceeded_flag


def convert_to_openai_tool_strict(tool_dict) -> ChatCompletionToolParam:
    """
    将工具字典转换为严格符合 OpenAI SDK 类型要求的 ChatCompletionToolParam
    """
    input_schema = tool_dict.get('inputSchema', {})

    # 仅保留 OpenAI 支持的 JSON Schema 字段
    supported_schema_keys = {
        'type', 'properties', 'required', 'enum', 'items',
        'format', 'default', 'description', 'anyOf', 'allOf', 'oneOf'
    }

    parameters = {
        k: v for k, v in input_schema.items()
        if k in supported_schema_keys
    }

    # 确保基础结构完整
    parameters.setdefault('type', 'object')
    parameters.setdefault('properties', {})
    parameters.setdefault('required', [])

    tool_param = {
        "type": "function",
        "function": {
            "name": tool_dict.get('name', ''),
            "description": tool_dict.get('description', ''),
            "parameters": parameters
        }
    }

    return cast(ChatCompletionToolParam, tool_param)


