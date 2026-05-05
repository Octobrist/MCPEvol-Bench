import os
import requests
import concurrent.futures
import json
import time
import sys

sys.path.append('FastChat')
# from ollama import chat
# from ollama import ChatResponse
from openai import OpenAI
from typing import List, Dict, Optional

from FastChat.fastchat.model.model_adapter import get_conversation_template
from FastChat.fastchat.conversation import get_conv_template
from requests.exceptions import Timeout, ConnectionError


api_key = 'sk-lM5LaCUWUMUfmcTcX47YDit9YOtXnIm19DAZPo5pILZx8U4f'
# api_key = 'sk-6dXGHo7A8AmKGRNUSRgqlaNhoRW6Hge5kzJUWXz55ctyuS5T'
base_url = 'https://api.huiyan-ai.cn/v1'

def chat_with_openai(
    conversation: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    调用 OpenAI 兼容 API 进行对话，支持自定义 api_key 和 base_url。

    参数:
        conversation: OpenAI 风格的消息列表，如 [{"role": "user", "content": "..."}]
        api_key: OpenAI API 密钥。若为 None，则从环境变量 OPENAI_API_KEY 读取。
        base_url: API 的基础 URL（例如 "https://api.openai.com/v1" 或本地代理地址）。
                  若为 None，则使用 OpenAI 默认端点。
        model: 模型名称。
        temperature: 生成随机性。
        max_tokens: 最大生成长度。

    返回:
        模型生成的回复字符串。
    """
    # 创建客户端，base_url 可为 None（使用默认）
    client = OpenAI(api_key=api_key, base_url=base_url)

    # os.environ['HTTP_PROXY'] = '99.72.0.200:3138'
    # os.environ['HTTPS_PROXY'] = '99.72.0.200:3138'
    # os.environ['HTTPS_PROXY'] = '12.8.3.205:3138'
    try:
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=2*60,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e: # tool_name
        return None


OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gpt-oss:120b"

# ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEME+c41Jc2J7sQRChnqj+5A8F80LPg4W5r4Zb2M4u35
# export OLLAMA_MODELS=/GLOBALFS/nudt_dwfeng_1/.bihu/qpf/chat/model
# ollama serve
# def call_ollama(prompt: str, model: str=DEFAULT_MODEL, timeout: int = 120) -> str:
#     try:
#         response: ChatResponse = chat(model=model, messages=[
#             {
#                 'role': 'user',
#                 'content': prompt,
#             },
#         ])
#
#         return response["message"]["content"]
#     except Exception as e:
#         return f"ERROR: {str(e)}"

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

def get_oss_prompt(conv):
    template = '''<|start|>system<|message|>{system_prompt}<|end|>\n
<|start|>developer<|message|><|end|>\n
<|start|>user<|message|>{user_prompt}<|end|>\n
<|start|>assistant'''
    return template.format(system_prompt=conv.system_message, user_prompt=conv.messages[0][1])

def format_oss_response(text: str):
    text_list = text.split("assistantfinal")
    return text_list[0], text_list[1]

def fschat_instruct_conv(conv, model, temperature, max_new_tokens, n, stop) -> list:
    worker_addr = get_worker_address(model)
    # conv = get_conv_template('phi3')
    # conv.append_message(conv.roles[0], prompt)
    # conv.append_message(conv.roles[1], None)
    if 'oss' in model:
        prompt = get_oss_prompt(conv)
    else:
        prompt = conv.get_prompt()

    gen_params = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": 1.0,
        "max_new_tokens": max_new_tokens,
        "stop": stop,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    response_list = []
    for _ in range(n):
        response = get_response(worker_addr, gen_params)
        # response = response.replace('\nAction: ', "")
        # response = response.replace('Action: ', "")
        # response = response.replace('\n', "")
        if 'oss' in model:
            response = format_oss_response(response)[1]
        response_list.append(response)
    return response_list

# <|channel|>analysis<|message|>Got weather data. Temp is 15C, sunny. Ready to answer.<|end|> <|start|>assistant<|channel|>final<|message|>今天旧金山阳光明媚，气温为 15 摄氏度。<|return|>

def transform_to_openai_messages(messages):
    new_messages = []
    for conv in messages:
        if 'user' in conv[0]:
            new_messages.append({'role': 'user', 'content': conv[1]})
        else:
            new_messages.append({'role': 'assistant', 'content': conv[1]})
    return new_messages

def continue_gen(model_name, messages, gen_continue=False) -> str:
    controller_addr = 'http://0.0.0.0:21001'
    gen_params = {
        "model": model_name,
        "temperature": 1.0,
        "max_new_tokens": 512,
        "echo": False,
        "top_p": 0.9,  # 0.9
    }
    # conv = get_conversation_template(os.path.join("LLaMA-Factory/lora", self.model_name))
    conv = get_conversation_template('qwen')
    for history_item in messages:
        role = history_item[0]
        content = history_item[1]
        if "user" in role:
            conv.append_message(conv.roles[0], content)
        elif "assistant" in role:
            conv.append_message(conv.roles[1], content)
        else:
            raise ValueError(f"Unknown role: {role}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # if not get_probs and not gen_continue:
    #     _add_to_set("Action: ",new_stop)
    # else:
    if gen_continue:
        prompt = prompt[:-len(f"<|im_end|>\n<|im_start|>assistant\n")] + '\n'
    gen_params.update(
        {
            "stop": ['\n'],
            "prompt": prompt,
        }
    )
    worker_addr = get_worker_address(model_name, controller_addr)
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

# if __name__ == "__main__":
#     from ollama import chat
#     from ollama import ChatResponse
#
#     response: ChatResponse = chat(model='gpt-oss:120b', messages=[
#         {
#             'role': 'user',
#             'content': '你是谁？',
#         },
#     ])
#     print(response['message']['content'])

    # 或者直接访问响应对象的字段
    # print(response.message.content)