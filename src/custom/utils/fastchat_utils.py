import argparse
import copy
import json
import os
import re
import time
import requests
import sys
sys.path.append("FastChat/")

from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
from requests.exceptions import Timeout, ConnectionError

from custom.utils.fschat_object import *


def build_simple_system_prompt(base_instruction: str, tools: list) -> str:
    """
    极简版：直接将工具列表转为字符串，按序号排列，嵌入 System Prompt。
    """
    # 1. 将工具列表转换为带序号的字符串
    # 使用 json.dumps 确保格式清晰，indent=2 增加可读性（可选，若需节省token可去掉）
    tool_str_parts = []
    for i, tool in enumerate(tools, 1):
        # 提取核心信息，避免过长的 schema 占用太多 token (可选优化)
        # 如果完全不在意长度，可以直接 str(tool)
        tool_info = json.dumps(tool, ensure_ascii=False, indent=2)
        tool_str_parts.append(f"Tool {i}:\n{tool_info}")

    tools_section = "\n\n".join(tool_str_parts)

    # 2. 定义输出和解析格式的简单说明
    format_instruction = """
## Output Format
To call a tool, output a JSON block like this:
```json
{
  "tool_calls": [
    {
      "function": {
        "name": "tool_name_here",
        "arguments": {
          "param1": "value1",
          "param2": value2
        }
      }
    }
  ]
}
```
If you feel that you have completed the task, output [FINISH] and add the your task execution summary.
"""
    final_prompt = f"""{base_instruction}
    
{tools_section}

{format_instruction}"""
    return final_prompt


def parse_llm_output_to_message(llm_text: str) -> AssistantMessage:
    """
    将 LLM 输出的文本字符串解析为 AssistantMessage 对象。

    支持格式:
    1. Markdown JSON block: ```json { ... } ```
    2. Raw JSON: { ... }
    3. Plain text: (视为普通回复)
    """
    if not llm_text or not llm_text.strip():
        return AssistantMessage(content="")
    if '[FINISH]' in llm_text:
        return AssistantMessage(content=llm_text)
    json_str = None

    # 1. 尝试提取 Markdown JSON 块
    match = re.search(r'```json\s*(.*?)\s*```', llm_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 2. 尝试提取最外层 JSON 对象
        start_idx = llm_text.find('{')
        end_idx = llm_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = llm_text[start_idx: end_idx + 1]

    # 3. 解析 JSON 并构建 Tool Calls
    if json_str:
        try:
            data = json.loads(json_str)
            # 检查是否包含标准的 tool_calls 结构
            if isinstance(data, dict) and 'tool_calls' in data and isinstance(data['tool_calls'], list):
                parsed_tool_calls = []
                for tc_item in data['tool_calls']:
                    # 提取 ID，如果没有则生成
                    tc_id = tc_item.get('id', str(uuid.uuid4()))
                    func_data = tc_item.get('function', {})
                    func_name = func_data.get('name', '')
                    func_args = func_data.get('arguments', {})
                    # 关键处理：确保 arguments 是 JSON 字符串
                    # 如果模型直接输出了 Dict，我们需要将其序列化回字符串
                    if isinstance(func_args, dict):
                        args_str = json.dumps(func_args, ensure_ascii=False)
                    elif isinstance(func_args, str):
                        # 如果已经是字符串，尝试验证其是否为合法 JSON，防止错误
                        args_str = func_args
                    else:
                        args_str = str(func_args)

                    parsed_tool_calls.append(ToolCall(
                        id=tc_id,
                        function=FunctionCall(name=func_name, arguments=args_str)
                    ))
                # 成功解析出工具调用，返回带有 tool_calls 的对象
                # 此时 content 通常为 None，因为工具调用指令本身不包含对话内容
                return AssistantMessage(content='', tool_calls=parsed_tool_calls)

        except json.JSONDecodeError as e:
            # JSON 解析失败，记录日志或忽略，回退到纯文本模式
            print(f"Warning: Failed to parse JSON from LLM output: {e}")
            llm_text = 'ERROR: Failed to parse JSON from LLM output: ' + llm_text
            pass
    # 4. 回退方案：视为普通文本回复
    return AssistantMessage(content=llm_text, tool_calls=None)

def parse_tool_calls_from_model_output(model_output: str):
    if not model_output:
        return None
    json_match = re.search(r'```json\s*(.*?)\s*```', model_output, re.DOTALL)

    json_str = ""
    if json_match:
        json_str = json_match.group(1)
    else:
        # 如果没有 markdown 标记，尝试直接解析整个字符串是否为 JSON
        # 或者查找第一个 { 和最后一个 }
        start_idx = model_output.find('{')
        end_idx = model_output.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = model_output[start_idx: end_idx + 1]
        else:
            return None

    try:
        data = json.loads(json_str)
        # 兼容两种结构：直接是 list，或者包含在 {"tool_calls": [...]} 中
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'tool_calls' in data:
            return data['tool_calls']
        else:
            return None
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from model output: {json_str[:100]}...")
        return None

def get_worker_address(model_name, controller_address="http://0.0.0.0:21001"):
    controller_addr = controller_address
    ret = requests.post(controller_addr + "/refresh_all_workers")
    ret = requests.post(controller_addr + "/list_models")
    models = ret.json()["models"]
    models.sort()
    # print(f"Use Model: {model_name} from {models}")

    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]

    if worker_addr == "":
        print(f"No available workers for {model_name}")
        raise ValueError
    return worker_addr

def get_response(worker_addr, gen_params):
    headers = {"User-Agent": "FastChat Client"}
    for _ in range(3):
        try:
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                headers=headers,
                json=gen_params,
                stream=True,
                timeout=120,
            )
            text = ""
            for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if line:
                    data = json.loads(line)
                    if data["error_code"] != 0:
                        assert False, data["text"]
                    text = data["text"]
            return text
        # if timeout or connection error, retry
        except Timeout:
            print("Timeout, retrying...")
        except ConnectionError:
            print("Connection error, retrying...")
        time.sleep(5)
    else:
        raise Exception("Timeout after 3 retries.")

def fschat_instruct_conv(conv, model, temperature, max_new_tokens, n, stop) -> list:
    worker_addr = get_worker_address(model)
    prompt = conv.get_prompt()

    gen_params = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": 0.9,
        "max_new_tokens": max_new_tokens,
        "stop": stop,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    response_list = []
    for _ in range(n):
        response = get_response(worker_addr, gen_params)
        response_list.append(response)
    return response_list


def convert_openai_trajectory_to_fastchat(trajectory_messages):
    """
    将包含 tool_calls 和 tool responses 的 OpenAI 轨迹消息
    转换为 FastChat 兼容的扁平化消息列表。

    策略:
    1. System -> System
    2. User -> User
    3. Assistant (with tool_calls) -> Assistant (内容为序列化的 JSON 工具调用)
    4. Tool -> User (内容为格式化的工具执行结果，模拟环境反馈)
    """
    converted_messages = []

    for msg in trajectory_messages:
        role = msg['role']
        content = msg.get('content', '')

        if role == 'system':
            # 系统提示直接保留
            converted_messages.append({
                "role": "system",
                "content": content
            })

        elif role == 'user':
            # 用户指令直接保留
            converted_messages.append({
                "role": "user",
                "content": content
            })

        elif role == 'assistant':
            if 'tool_calls' in msg and msg['tool_calls']:
                # 如果助手发起了工具调用，将其序列化为字符串
                # 使用 JSON 格式以便模型解析参数
                tool_calls_json = json.dumps(msg['tool_calls'], ensure_ascii=False, indent=2)

                # 可选：添加标签帮助模型识别这是工具调用而非普通文本
                # Qwen/Vicuna 通常能理解这种结构
                formatted_content = f"Action: Call Tools\n{tool_calls_json}"

                converted_messages.append({
                    "role": "assistant",
                    "content": formatted_content
                })
            else:
                # 普通助手回复
                converted_messages.append({
                    "role": "assistant",
                    "content": content
                })

        elif role == 'tool':
            # 【关键修改】将 tool 角色转换为用户角色
            # 这模拟了“环境”或“系统”以用户身份向模型反馈执行结果
            tool_call_id = msg.get('tool_call_id', 'unknown')
            # 清理内容：如果内容是复杂的对象字符串（如你的示例中的 [TextContent(...)]），
            # 最好尝试提取其中的真实文本，或者保留原样让模型学习。
            # 这里我们保留原始内容，但加上明确的前缀。

            # 格式建议: "Observation: <result>" 或 "Tool Output [id]: <result>"
            # 这种格式在许多 ReAct 或 Function Call 微调数据集中很常见
            formatted_observation = f"Observation (for call {tool_call_id}):\n{content}"

            converted_messages.append({
                "role": "user",  # 这里改为 user
                "content": formatted_observation
            })

    return converted_messages


def chat_with_fastchat(this_messages, model_name, tools, temperature=0.7, max_new_tokens=512):
    """
    主函数：接收 OpenAI 格式消息（含 tool），转换为 FastChat 格式并推理。
    """
    messages = copy.deepcopy(this_messages)
    messages[0]['content'] = build_simple_system_prompt(this_messages[0]['content'], tools)
    # 1. 初始化对话模板
    if 'qwen' in model_name or 'Qwen' in model_name:
        conv_name = 'qwen-2.5-7b-instruct'
    elif 'gemma' in model_name:
        conv_name = 'gemma-4-31b-it'
    elif 'llama' in model_name:
        conv_name = 'llama-3'
    else:
        raise NotImplementedError
    conv = get_conv_template(conv_name)
    conv.messages = []  # 清空历史

    # 2. 预处理：检查是否包含工具交互，如果有则进行转换
    # 判断标准：存在 role=='tool' 或 assistant 中有 tool_calls
    has_tools = any(m['role'] == 'tool' or 'tool_calls' in m for m in messages)

    if has_tools:
        processed_messages = convert_openai_trajectory_to_fastchat(messages)
    else:
        processed_messages = messages

    # 3. 将处理后的消息填入 Conv 模板
    for msg in processed_messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            conv.set_system_message(content)
        elif role == 'user':
            # 对应 conv.roles[0]，通常是 "USER" 或 "Human"
            conv.append_message(conv.roles[0], content)
        elif role == 'assistant':
            # 对应 conv.roles[1]，通常是 "ASSISTANT" 或 "Assistant"
            conv.append_message(conv.roles[1], content)
    if conv_name == 'qwen-2.5-7b-instruct':
        conv.messages[-1][1] += '/no_think'
    # 4. 添加生成占位符 (关键步骤：告诉模型该你说话了)
    conv.append_message(conv.roles[1], None)

    # 5. 调用推理
    # 假设 fschat_instruct_conv 是你环境中已有的推理函数
    try:
        results = fschat_instruct_conv(conv, model_name, temperature, max_new_tokens, n=1, stop=[])
        if not results:
            return AssistantMessage(content=f"ERROR in chat_with_fastchat: empty result")
        output = parse_llm_output_to_message(results[0])
        return output
    except Exception as e:
        return AssistantMessage(content=f"ERROR in chat_with_fastchat: {str(e)}")

if __name__ == "__main__":
    conv_name = 'qwen-2.5-7b-instruct'
    conv = get_conv_template(conv_name)
    conv.messages = []
    conv.append_message(conv.roles[0], 'hi')  #/no_think
    conv.append_message(conv.roles[1], None)
    model_responses = fschat_instruct_conv(conv, 'qwen-gen-0', 1.0, 512, 1, '')
    for i, res in enumerate(model_responses):
        print(f"[Result {i + 1}]:\n{res}\n")
