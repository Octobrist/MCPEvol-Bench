import json
import copy
import re
import xml.etree.ElementTree as ET

from collections import defaultdict
from typing import List, Dict, Union, Optional, Any

def get_1mcp_server_tools(t_status, mcp_server_name):
    server_tools = []
    server_name_map = {
        'server-sequential-thinking': '@modelcontextprotocol/server-sequential-thinking',
        'mcp-server-time': '@guanxiong/mcp-server-time',
        'server-memory': '@guanxiong/server-memory',
        'pollinations/model-context-protocol': '@pollinations/model-context-protocol'
    }
    if mcp_server_name in server_name_map:
        mcp_server_name = server_name_map[mcp_server_name]
    for tool in t_status.tools:
        if tool.name.startswith(f'{mcp_server_name}_1mcp_'):
            cp_tool = copy.deepcopy(tool)
            cp_tool.name = tool.name.replace(f'{mcp_server_name}_1mcp_', '')
            server_tools.append(cp_tool)
    return server_tools


def get_1mcp_server_by_tool(t_status, tool_name):
    servers = []
    for tool in t_status.tools:
        if f'_1mcp_{tool_name}' in tool.name:
            server_name = tool.name.replace(f'_1mcp_{tool_name}', '')
            servers.append(server_name)
    return servers

def exist_tool_calls(traj):
    for item in traj:
        if item['role'] == 'tool':
            return True
    return False

def clean_all_test_cases(cases):
    with open(f'custom/data/test_cases/chosen_idx.json', 'r', encoding='utf-8') as f:
        real_test_cases = json.load(f)
    real_test_case_ids = [item['idx'] for item in real_test_cases['samples']]
    cases = [
        case
        for case in cases
        if case['idx'] in real_test_case_ids
    ]
    return cases

def get_server_npx_infos(servers, data_source='github'):
    npx_name_set = set()
    npx_infos = {}
    if data_source == 'github':
        for category, items in servers.items():
            for item_idx, item in enumerate(items['item']):
                if 'command' in item.keys() and 'extracted' in item['command'].keys():
                    extracted_infos = item['command']['extracted']
                    if not isinstance(extracted_infos, list):
                        continue
                    for ex_info in extracted_infos:
                        if 'npm_info' in ex_info.keys():
                            for npx_info in ex_info['npm_info']:
                                npx_name = list(npx_info.keys())[0]
                                npx_infos[npx_name] = npx_info[npx_name]

    elif data_source == 'modelscope':
        for category, items in servers.items():
            for item_idx, item in enumerate(items):
                if 'npm_info' in item.keys():
                    for npx_info in item['npm_info']:
                        npx_name = list(npx_info.keys())[0]
                        npx_infos[npx_name] = npx_info[npx_name]

    elif data_source == 'smithery':
        for item in servers:
            if 'npm_info' in item.keys():
                for npx_info in item['npm_info']:
                    npx_name = list(npx_info.keys())[0]
                    npx_infos[npx_name] = npx_info[npx_name]
                    npx_name_set.add(npx_name)

    elif data_source == 'npx':
        for npx_name, value in servers.items():
            if 'npm_info' in value.keys() and len(value['npm_info']) > 0:
                npx_infos[npx_name] = value['npm_info'][0][npx_name]
            # else:
                # print(npx_name)

    return npx_infos

def load_all_npx_infos(prefix_path:str, json_paths: Dict[str, str]):
    all_npx_infos = {}
    for data_source, json_path in json_paths.items():
        with open(f'{prefix_path}/{json_path}', 'r', encoding='utf-8') as f:
            raw_dict = json.load(f)
        all_npx_infos.update(get_server_npx_infos(raw_dict, data_source))
    return all_npx_infos

def load_all_npx_tools(json_paths: List[str]):
    """
    加载多个 JSON 文件，并对其中的数据进行去重。

    假设每个 JSON 文件的内容是：
      - 一个 JSON 对象（dict）
      - 或一个 JSON 数组（list）
      - 或其他 JSON 值（如字符串、数字等）

    所有顶层元素（如果是 list，则展开；如果是单个值，则作为一项）会被收集，
    然后基于 JSON 标准化字符串进行语义去重。

    Args:
        json_paths: JSON 文件路径列表

    Returns:
        去重后的数据列表（保持原始数据类型，如 dict, list, str 等）
    """
    all_items = []

    for path in json_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found - {path}")
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(path, len(data))
            for item in data:
                if 'tool_infos' in item.keys():
                    if item['tool_infos'] != []:
                        all_items.append(item)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {path}: {e}")
            continue

    # 去重（基于完整内容）
    seen = set()
    unique_items = []
    for item in all_items:
        if not isinstance(item, dict):
            print(f"Skipping non-dict item: {item}")
            continue
        # 检查必要字段
        if 'npx_name' not in item or 'npx_version' not in item:
            print(f"Skipping item missing 'npx_name' or 'npx_version': {item}")
            continue
        try:
            canonical = json.dumps(item, sort_keys=True, ensure_ascii=False)
        except TypeError:
            continue
        if canonical not in seen:
            seen.add(canonical)
            unique_items.append(item)

    # 第二步：按 npx_name -> npx_version 分组
    grouped: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))

    for item in unique_items:
        name = item['npx_name']
        myversion = item['npx_version']
        grouped[name][myversion].append(item)

    # 转为普通 dict（便于序列化）
    return {
        name: dict(versions)
        for name, versions in grouped.items()
    }

def analyze_versions(version_list: List[str]) -> Dict[str, str]:
    """
    对版本号列表进行统计分析，返回最小、下四分位、中位数、上四分位、最大版本。

    使用语义化版本（SemVer）规则进行排序和比较。

    Args:
        version_list: 版本号字符串列表，如 ['0.0.1', '1.0.0', '2.1.0']

    Returns:
        字典，包含：
        {
            'min': '0.0.1',
            'q1':  '0.5.0',   # 第一四分位数（25%）
            'median': '1.0.0', # 中位数（50%）
            'q3':  '1.5.0',   # 第三四分位数（75%）
            'max': '2.0.0'
        }

    Raises:
        ValueError: 如果输入列表为空
    """
    if not version_list:
        raise ValueError("版本列表不能为空")

    # 去重并解析为 Version 对象
    # unique_versions = list(set(version_list))  # 先去重
    # try:
    #     parsed = [parse(v) for v in unique_versions]
    #     parsed.sort()  # 按 SemVer 规则排序
    # except:
    #     parsed = unique_versions
    parsed = version_list
    n = len(parsed)
    result = {}

    # 最小 & 最大
    result['min'] = str(parsed[0])
    result['max'] = str(parsed[-1])

    # 中位数 (50%)
    if n % 2 == 1:
        median = parsed[n // 2]
    else:
        median = parsed[n // 2 - 1]  # 或取平均？但版本无法平均，通常取 lower median
    result['median'] = str(median)

    # 四分位数：使用 inclusive method (Tukey's hinges) 或 linear interpolation
    # 这里采用简单方法：基于索引位置（与 numpy.percentile 默认一致）
    def get_percentile(percentile: float) -> str:
        # percentile: 0.25 for Q1, 0.75 for Q3
        pos = (n - 1) * percentile
        if pos.is_integer():
            return str(parsed[int(pos)])
        else:
            lower = parsed[int(pos)]
            upper = parsed[int(pos) + 1]
            # 由于版本不能插值，通常取 lower（保守）或按实际需求
            # 这里我们取 lower（与许多统计实践一致，尤其离散数据）
            return str(lower)

    result['q1'] = get_percentile(0.25)
    result['q3'] = get_percentile(0.75)

    return result

def filter_packages_with_at_least_k_versions(grouped_data, k):
    filtered = {}
    for pkg_name, version_dict in grouped_data.items():
        if len(version_dict) >= k:
            filtered[pkg_name] = version_dict
    return filtered

def get_npx_total_version_number(npx_dict):
    total_number = 0
    for k, v in npx_dict.items():
        total_number += len(v)
    return total_number

def extract_xml_from_markdown(text: str) -> str:
    """
    从 Markdown 文本中提取第一个被 ```xml ... ``` 包裹的 XML 内容。
    如果没有找到，则返回原始字符串。

    参数:
        text (str): 输入的 Markdown 文本

    返回:
        str: 提取的 XML 内容（去首尾空白），或原始 text（若未匹配）
    """
    # 匹配 ```xml ... ``` 代码块（支持大小写和空格）
    pattern = r"```(?:\s*xml\s*)(.*?)```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    else:
        return text  # 未匹配到，返回原字符串

def parse_single_mcp_tool_use_xml(xml_string: str) -> Dict:
    """
    Parse the XML response generated by the MCP tool-use prompt.

    Supports both:
      - Single tool: <target_tool_call> (singular)
      - Multi tool: <target_tool_calls> (plural, with multiple <call> elements)

    Returns a dictionary with:
      - 'server_analysis': str
      - 'question': str
      - 'tool_calls': List[Dict]  # each has 'tool_name' and 'parameters' (dict)
    """
    xml_string = extract_xml_from_markdown(xml_string.strip())
    try:
        root = ET.fromstring(xml_string.strip())
    except ET.ParseError as e:
        # print(f"Invalid XML: {e}")
        return {}

    # Extract common fields
    server_analysis_elem = root.find("server_analysis")
    question_elem = root.find("question")

    server_analysis = (server_analysis_elem.text or "").strip() if server_analysis_elem is not None else ""
    question = (question_elem.text or "").strip() if question_elem is not None else ""

    tool_calls = []

    # Try multi-tool format: <target_tool_calls> containing multiple <call> blocks
    target_tool_calls_elem = root.find("target_tool_calls")
    if target_tool_calls_elem is not None:
        # In multi-tool, each call is wrapped in an element (e.g., <call> or direct children)
        # Based on your spec, we use anonymous child blocks (no wrapper name), so iterate all children
        for call_elem in target_tool_calls_elem:
            if call_elem.tag == 'target_tool_call':
                # fallback support: sometimes singular inside plural (be flexible)
                pass
            tool_name_elem = call_elem.find("tool_name")
            tool_name = (tool_name_elem.text or "").strip() if tool_name_elem is not None else ""

            params = {}
            params_elem = call_elem.find("parameters")
            if params_elem is not None:
                for param in params_elem.findall("parameter"):
                    name = param.get("name")
                    value = (param.text or "").strip()
                    if name:
                        # Optional: try to auto-convert common types (bool, int, float)
                        # Here we keep as string for safety unless you need typing
                        params[name] = value
            tool_calls.append({
                "tool_name": tool_name,
                "parameters": params
            })
    else:
        # Try single-tool format: <target_tool_call>
        target_tool_call_elem = root.find("target_tool_call")
        if target_tool_call_elem is not None:
            tool_name_elem = target_tool_call_elem.find("tool_name")
            tool_name = (tool_name_elem.text or "").strip() if tool_name_elem is not None else ""

            params = {}
            params_elem = target_tool_call_elem.find("parameters")
            if params_elem is not None:
                for param in params_elem.findall("parameter"):
                    name = param.get("name")
                    value = (param.text or "").strip()
                    if name:
                        params[name] = value
            tool_calls.append({
                "tool_name": tool_name,
                "parameters": params
            })

    return {
        "server_analysis": server_analysis,
        "question": question,
        "tool_calls": tool_calls
    }

def parse_mutil_mcp_tool_use_xml(xml_string: str) -> Dict:
    """
    Parse the XML response generated by the updated MCP tool-use prompt.

    Now supports only the simplified format:
      - <target_tools> containing multiple <tool_name> elements (no parameters)

    Returns a dictionary with:
      - 'server_analysis': str
      - 'question': str
      - 'tool_calls': List[Dict]  # each has 'tool_name' and 'parameters' = {}
    """
    xml_string = extract_xml_from_markdown(xml_string.strip())
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        # print(f"Invalid XML: {e}")
        return {}

    # Extract common fields
    server_analysis_elem = root.find("workflow_analysis")
    question_elem = root.find("question")

    server_analysis = (server_analysis_elem.text or "").strip() if server_analysis_elem is not None else ""
    question = (question_elem.text or "").strip() if question_elem is not None else ""

    tool_calls = []
    # 查找 <target_servers> 元素
    target_servers_elem = root.find("target_servers")
    if target_servers_elem is not None:
        # 遍历每个 <server>
        for server_elem in target_servers_elem.findall("server"):
            # 可选：获取 server name（如果后续需要）
            # server_name = server_elem.get("name")

            # 遍历该 server 下的所有 <tool>
            for tool_elem in server_elem.findall("tool"):
                tool_name = (tool_elem.text or "").strip()
                if tool_name:  # 只添加非空工具名
                    tool_calls.append({
                        "tool_name": tool_name,
                        "parameters": {}  # 当前无参数
                    })

    return {
        "server_analysis": server_analysis,
        "question": question,
        "tool_calls": tool_calls
    }


def parse_string_to_json(markdown_text: str):
    """
    从可能包含 ```json ... ``` 的字符串中提取 JSON 内容，并保存为文件。

    参数:
        markdown_text (str): 包含 JSON 的字符串（可能被 ```json 和 ``` 包裹）
        output_file (str): 要保存的 JSON 文件路径（例如 "output.json"）
    """
    # 使用正则表达式提取 ```json ... ``` 中的内容
    match = re.search(r'```(?:json)?\s*({.*?})\s*```', markdown_text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
    else:
        # 如果没有找到代码块，尝试直接解析整个字符串
        json_str = markdown_text.strip()

    # 解析 JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析 JSON 内容: {e}")

    return data

def format_tools_to_string(tools, dict_flag=True):
    new_tools = []
    for idx, tool in enumerate(tools):
        tool_new_dict = {}
        if dict_flag:
            name = tool['name']
            desc = tool['description']
            schema = tool.get('inputSchema', {})
            output_schema = tool.get('outputSchema', {})
        else:
            tool_new_dict = {}
            name = tool.name
            desc = tool.description
            schema = tool.inputSchema
            output_schema = tool.outputSchema
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        tool_new_dict['name'] = name
        tool_new_dict['description'] = desc
        tool_new_dict['properties'] = properties
        tool_new_dict['required'] = required
        tool_new_dict['outputSchema'] = str(output_schema)
        new_tools.append(tool_new_dict)
    return json.dumps(new_tools, indent=2, ensure_ascii=False)

def load_all_json_files(directory: str):
    """
    加载指定目录下所有 .json 文件（不递归子目录）。
    返回一个字典：{文件名: 解析后的 JSON 数据}
    """
    json_data = {}
    path = Path(directory)

    if not path.is_dir():
        raise ValueError(f"路径不是有效目录: {directory}")

    for json_file in path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data[json_file.name] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ 跳过无效 JSON 文件: {json_file} - 错误: {e}")
        except Exception as e:
            print(f"❌ 读取文件失败: {json_file} - 错误: {e}")

    return json_data

def convert_params(raw_params, param_descriptions):
    """
    将字符串参数转换为对应类型的参数

    Args:
        raw_params (dict): 原始参数字典，值都是字符串
        param_descriptions (dict): 参数描述字典，包含类型信息

    Returns:
        dict: 转换后的参数字典
    """
    converted_params = {}

    for param_name, param_value in raw_params.items():
        # 如果参数描述中没有这个参数，保持原样
        if param_name not in param_descriptions:
            converted_params[param_name] = param_value
            continue

        param_desc = param_descriptions[param_name]
        param_type = param_desc.get('type', 'string')

        try:
            if param_type == 'boolean':
                # 处理布尔类型，支持多种字符串表示
                if isinstance(param_value, str):
                    lower_value = param_value.lower()
                    if lower_value in ['true', '1', 'yes', 'on']:
                        converted_value = True
                    elif lower_value in ['false', '0', 'no', 'off']:
                        converted_value = False
                    else:
                        # 无法识别的布尔字符串，使用默认逻辑
                        converted_value = bool(param_value)
                else:
                    converted_value = bool(param_value)

            elif param_type == 'integer':
                converted_value = int(param_value)

            elif param_type == 'number':
                converted_value = float(param_value)

            elif param_type == 'array':
                # 尝试解析JSON数组，如果失败则按逗号分割
                if isinstance(param_value, str):
                    try:
                        converted_value = json.loads(param_value)
                    except (json.JSONDecodeError, TypeError):
                        converted_value = [item.strip() for item in param_value.split(',') if item.strip()]
                else:
                    converted_value = param_value

            elif param_type == 'object':
                # 尝试解析JSON对象
                if isinstance(param_value, str):
                    try:
                        converted_value = json.loads(param_value)
                    except (json.JSONDecodeError, TypeError):
                        converted_value = param_value
                else:
                    converted_value = param_value

            else:  # string 或其他类型
                converted_value = param_value

            converted_params[param_name] = converted_value

        except (ValueError, TypeError) as e:
            # 转换失败时保持原样并警告
            # print(f"Warning: Failed to convert {param_name}={param_value} to {param_type}: {e}")
            converted_params[param_name] = param_value

    return converted_params


# !/usr/bin/env python3
# directory_tree_simple.py

import os
from pathlib import Path


def generate_tree_string(root_path, exclude_patterns=None, max_depth=None):
    """
    生成纯文本目录树字符串（专为大模型优化）

    特点：
    - 仅用空格缩进表示层级（每级2空格）
    - 目录以 / 结尾（如: src/）
    - 文件直接显示名称（如: main.py）
    - 无连接符（├──/└──）、无图标、无颜色
    - 自动排除开发垃圾文件

    返回：纯文本字符串
    """
    if exclude_patterns is None:
        exclude_patterns = {
            '.git', '__pycache__', '.venv', 'venv', 'node_modules',
            '.DS_Store', '.idea', '.vscode', '.env', 'dist', 'build',
            '.pytest_cache', '.mypy_cache', 'Thumbs.db'
        }

    root = Path(root_path).resolve()
    if not root.exists() or not root.is_dir():
        return f"Error: Invalid directory path - {root_path}"

    def _build_tree(path, prefix="", depth=0):
        if max_depth is not None and depth > max_depth:
            return []

        try:
            entries = [
                e for e in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                if e.name not in exclude_patterns
                   and not e.name.startswith('.')  # 排除隐藏文件
                   and not e.name.startswith('__')  # 排除 __pycache__ 等
            ]
        except PermissionError:
            return []

        lines = []
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            current_prefix = "  " * depth  # 每级2空格缩进

            if entry.is_dir():
                lines.append(f"{current_prefix}{entry.name}/")
                lines.extend(_build_tree(entry, prefix, depth + 1))
            else:
                lines.append(f"{current_prefix}{entry.name}")

        return lines

    # 构建树（根目录单独处理）
    tree_lines = [f"{root.name}/"]
    tree_lines.extend(_build_tree(root, depth=1))

    return "\n".join(tree_lines)

from datetime import datetime, timedelta


def offset_date(days: int) -> str:
    """返回偏移后的日期字符串，格式为：YYYY-MM-DD (星期X)"""
    target_date = datetime.now() + timedelta(days=days)
    date_str = target_date.strftime("%Y-%m-%d")
    # 获取星期几 (0=Monday, 6=Sunday)
    weekday_map = {
        0: "星期一", 1: "星期二", 2: "星期三",
        3: "星期四", 4: "星期五", 5: "星期六", 6: "星期日"
    }
    weekday = weekday_map[target_date.weekday()]

    return f"{date_str} ({weekday})"

def return_mcp_description(mcp_server_name, simple_name, npx_infos, live_infos):
    if mcp_server_name in npx_infos.keys() and 'description' in npx_infos[mcp_server_name].keys():
        return npx_infos[mcp_server_name]['description']

    for item_info in live_infos:
        if mcp_server_name in str(item_info['config']) or simple_name in str(item_info['config']):
            return item_info['description']
    return None

