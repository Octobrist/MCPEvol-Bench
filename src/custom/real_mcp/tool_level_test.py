import copy
import os
import json
import random
import sys

sys.path.append('/path/to/FastChat')
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from custom.prompt.test_case_prompt import MULTI_SERVER_MULTI_TOOL_PROMPT
from custom.utils.test_utils import *
from custom.utils.openai_llm import chat_with_openai
from custom.utils.mcp_utils import *
from custom.agents.SimpleOpenAIAgent import ChatModel


def get_all_evolution_status(mutation_path):
    all_evolution_status = []
    while True:
        if 'HYBRID' in mutation_path:
            with open(os.path.join(mutation_path, 'evolution.json'), 'r', encoding='utf-8') as __f:
                all_evolution_status.append(json.load(__f))
            mutation_path = mutation_path.rsplit('HYBRID', 1)[0]
        else:
            break
    return all_evolution_status

def clean_test_case(_data: dict) -> dict:
    """
    清洗测试用例字典，只保留 question, servers, tool_calls
    """
    allowed_keys = {"question", "servers", "tool_calls"}
    return {k: v for k, v in _data.items() if k in allowed_keys}

def change_test_case_servers(_test_case, args):
    if args.mutation_type == 'DESC' or args.mutation_type == 'PARAM':
        return _test_case
    new_test_case = copy.deepcopy(_test_case)

    with open(args.mutation_path+'/evolution.json', 'r', encoding='utf-8') as f:
       _evolutions = json.load(f)

    for _server_name, _tool_list in _test_case['servers'].items():
        if _server_name in _evolutions.keys():
            if _evolutions[_server_name]['mutation_type'] != 'TOOL':
                return _test_case
            mutation_tools = get_1mcp_server_tools(tool_status, _server_name)
            mutation_tool_names = [tool.name for tool in mutation_tools]
            modified_tool_name = None
            if 'new_tool_config' in _evolutions[_server_name].keys():
                modified_tool_name = _evolutions[_server_name]['new_tool_config']['name']
                new_test_case['servers'][_server_name].append(modified_tool_name)
            elif 'auto_optimizations' in _evolutions[_server_name].keys():
                modified_tool_name = _evolutions[_server_name]['new_tool_name']
                new_test_case['servers'][_server_name].append(modified_tool_name)
                for _optimization in _evolutions[_server_name]['auto_optimizations']:
                    new_test_case['servers'][_server_name].append(_optimization['tool_name'])
            elif 'deleted_tool_name' in _evolutions[_server_name].keys():
                if _evolutions[_server_name]['deleted_tool_name'] in new_test_case['servers'][_server_name]:
                    new_test_case['servers'][_server_name].remove(_evolutions[_server_name]['deleted_tool_name'])
                new_test_case['servers'][_server_name].append(_evolutions[_server_name]['migration_target_tool'])
                modified_tool_name = _evolutions[_server_name]['migration_target_tool']
            elif 'old_tool_name' in _evolutions[_server_name].keys():
                if _evolutions[_server_name]['old_tool_name'] in new_test_case['servers'][_server_name]:
                    new_test_case['servers'][_server_name].remove(_evolutions[_server_name]['old_tool_name'])
                new_test_case['servers'][_server_name].append(_evolutions[_server_name]['new_tool_config']['name'])
                modified_tool_name = _evolutions[_server_name]['new_tool_config']['name']
            else:
                raise ValueError('Unknown tool name')
            if modified_tool_name not in mutation_tool_names:
                print(_server_name, modified_tool_name)
    return new_test_case

def return_involved_tool_status(involved_server_and_tools, this_tool_status):
    involved_tool_status = []
    for _server_name in involved_server_and_tools.keys():
        _involved_tools = involved_server_and_tools[_server_name]
        exist_tools = get_1mcp_server_tools(this_tool_status, _server_name)
        for tool in exist_tools:
            if 'generate_chart' in tool.name:
                tool.inputSchema['properties']['datasets']['items']['properties']['data'] = {'type': 'array', 'items': {'type': 'number'},}
            if tool.name in _involved_tools:
                tool_dict = {
                    'name': tool.name,
                    'description': tool.description,
                    'inputSchema': tool.inputSchema,
                }
                involved_tool_status.append(tool_dict)
    return involved_tool_status

def _get_desc_tool_details(servers, m_path):
    """获取 DESC 变异涉及的工具详细定义"""
    try:
        with open(os.path.join(m_path, 'mutated_server_status.json')) as f:
            status = json.load(f)
    except Exception:
        raise FileNotFoundError

    details = []
    for srv, tools in servers.items():
        tool_list = status.get(srv, {}).get('tools')
        for t in tool_list:
            if t['name'] in tools:
                details.append({k: t.get(k) for k in ['name', 'description', 'inputSchema']})
    return details

def return_involved_mutated_tool_status(servers, original_status, args):
    _all_evolutions = get_all_evolution_status(args.mutation_path)

    mutated_tool_status = []
    for _server_name, _tool_list in servers.items():
        mutation_type_list = [_evolutions[_server_name]['mutation_type'] for _evolutions in _all_evolutions if _server_name in _evolutions]
        mutated_tool_status.extend(return_involved_tool_status({_server_name: _tool_list}, original_status))
        if 'DESC' in mutation_type_list:
            for _evolutions in _all_evolutions[::-1]: # 遍历evol
                if _server_name in _evolutions and _evolutions[_server_name]['mutation_type'] == 'DESC':
                    _tool_name = [k for k in _evolutions[_server_name].keys() if k not in ['mutation_type', 'task_idx']][0]
                    for tool_sta in mutated_tool_status:
                        if tool_sta['name'] != _tool_name:
                            continue
                        # 修改tool描述和参数描述
                        if 'tool' in _evolutions[_server_name][_tool_name].keys():
                            tool_sta['description'] = _evolutions[_server_name][_tool_name]['tool']['modified_description']
                        if 'parameters' in _evolutions[_server_name][_tool_name].keys():
                            for _param_dict in _evolutions[_server_name][_tool_name]['parameters']:
                                if _server_name == '@sylphlab/pdf-reader-mcp' and _tool_name == 'read_pdf' and _param_dict['parameter_name'] in ['path', 'url', 'pages']:
                                    tool_sta['inputSchema']['properties']['sources']['items']['properties'][_param_dict['parameter_name']]['description'] = _param_dict['modified_description']
                                elif _server_name == '@tsmztech/mcp-server-salesforce' and _tool_name == 'salesforce_search_all' and _param_dict['parameter_name'] in ['name', 'where', 'orderBy', 'limit']:
                                    tool_sta['inputSchema']['properties']['objects']['items']['properties'][_param_dict['parameter_name']]['description'] = _param_dict['modified_description']
                                elif _server_name == '@itseasy21/mcp-knowledge-graph' and _tool_name == 'create_relations' and _param_dict['parameter_name'] in ['from', 'to', 'relationType']:
                                    tool_sta['inputSchema']['properties']['relations']['items']['properties'][_param_dict['parameter_name']]['description'] = _param_dict['modified_description']
                                else:
                                    if _param_dict['parameter_name'] not in tool_sta['inputSchema']['properties']:
                                        print(_server_name, _tool_name, _param_dict['parameter_name'])
                                        continue
                                    tool_sta['inputSchema']['properties'][_param_dict['parameter_name']]['description'] = _param_dict['modified_description']
    return mutated_tool_status


def _is_server_involved(tool_list, evolution_info, m_type):
    """判断单个服务器是否涉及变异"""
    # 获取变异涉及的主要工具名 (取第一个key)
    mutated_tool = next(iter(evolution_info), None)
    if m_type in ('DESC', 'PARAM'):
        # DESC/PARAM: 检查变异工具是否在用
        return mutated_tool in tool_list if mutated_tool else False
    elif m_type == 'TOOL':
        # TOOL: 检查新增配置 或 被删除/旧工具是否在用
        if 'new_tool_config' in evolution_info or 'auto_optimizations' in evolution_info:
            return True
        removed = evolution_info.get('deleted_tool_name') or evolution_info.get('old_tool_name')
        return removed in tool_list if removed else False
    return False


def return_involved_mutated_test_cases(_test_cases, args):
    _all_evolutions = get_all_evolution_status(args.mutation_path)

    new_test_cases = []
    for _evolutions in _all_evolutions:
        for tc in _test_cases:
            for srv, tools in tc.get('servers', {}).items():
                if srv in _evolutions and _is_server_involved(tools, _evolutions[srv], _evolutions[srv]['mutation_type']):
                    if tc not in new_test_cases:
                        new_test_cases.append(tc)
                    break
    return new_test_cases

# 1
def build_mcp_classification_prompt(mcp_servers: dict) -> str:
    services_list = "\n".join([f"- {name}: {description}" for name, description in mcp_servers.items()])

    prompt = f"""You are a system architecture expert. Based on the following MCP server descriptions, define a set of high-level functional categories (e.g., Finance, File System, Browser, Authentication, etc.), assign each server to one or more categories, and provide a short description for each category.

# Requirements:
1. Define intuitive, meaningful category names that reflect broad functional domains.
2. The assigned categories should be specific to a particular field and not vague or multiple fields, must including these categories: Knowledge, Research, Development Tools, Media & Documentation, Data & Analytics and Business & Commerce.
3. Use **no more than 10 distinct categories** in total.
4. Aim to distribute the servers **as evenly as possible** across the categories.
5. Each server must be assigned to one category.
6. Output a valid JSON object with three keys:
   - "categories": a list of objects, each containing "name" and "description" fields;
   - "assignments": a mapping from each server name to a list of category names it belongs to;
   - The "description" for each category should be a concise sentence (10–20 words) explaining what kinds of services belong in it.

Example output structure:
{{
  "categories": [
    {{"name": "Finance", "description": "Services related to payments, billing, and financial transactions."}},
    {{"name": "FileSystem", "description": "Operating system files, documents, and other related content."}}
  ],
  "assignments": {{
    "payment-service": ["Finance"],
    "file-converter": ["FileSystem"]
  }}
}}

MCP servers to classify:
{services_list}
"""
    return prompt

# 2
def generate_combinations(server_list, num_combinations=100, min_size=3, max_size=6):
    """
    从 server_list 中生成多个随机组合

    Args:
        server_list: List[str] - 候选 server 列表
        num_combinations: int - 要生成多少个组合
        min_size: int - 每个组合最小长度
        max_size: int - 每个组合最大长度

    Returns:
        List[List[str]] - 多个组合的列表
    """
    if len(server_list) < min_size:
        raise ValueError(f"server_list 至少需要 {min_size} 个元素")

    combinations = []
    for _ in range(num_combinations):
        # 随机决定当前组合大小（2~5，但不超过列表长度）
        k = random.randint(min_size, min(max_size, len(server_list)))
        # 无放回抽样（组合内不重复）
        combo = random.sample(server_list, k)
        combinations.append(combo)

    return combinations

# 2
def generate_combination_descriptions(combinations, name_to_desc):
    """
    Generate human-readable English descriptions for server combinations.

    Args:
        combinations: List[List[str]] - e.g., [["dev-db", "prod-db"], ...]
        name_to_desc: Dict[str, str] - e.g., {"dev-db": "Development PostgreSQL database", ...}

    Returns:
        List[str] - Descriptive sentences for each combination
    """
    descriptions = []

    for combo in combinations:
        if not combo:
            descriptions.append("Empty combination.")
            continue

        # Build list of "Description (name)" strings
        items = []
        for server in combo:
            desc = name_to_desc.get(server, f"Unknown service ({server})")
            items.append(f"{desc} ({server})")

        # Natural English joining: "A, B, and C"
        if len(items) == 1:
            intro = items[0]
        elif len(items) == 2:
            intro = " and ".join(items)
        else:
            intro = ", ".join(items[:-1]) + ", and " + items[-1]

        full_desc = f"This combination includes: {intro}."
        descriptions.append(full_desc)

    return descriptions

# 2
def generate_multi_tool_descriptions(combinations, server_desc, tool_status):
    """
    Generate rich descriptions for servers with multiple tools.

    Args:
        combinations: List[List[str]] - e.g., [["dev-db", "prod-db"]]
        server_desc: Dict[str, str] - server purpose
        tool_status: all 1mcp tools

    Returns:
        List[str] - One detailed block per combination
    """
    descriptions = []

    for combo in combinations:
        if not combo:
            descriptions.append("Empty combination.")
            continue

        blocks = []
        for server in combo:
            # Server header
            srv_desc = server_desc.get(server, "")
            block = f"**{server}**: {srv_desc}"
            tools = get_1mcp_server_tools(tool_status, server)
            if tools:
                tools_desc = format_tools_to_string(tools, dict_flag=False)
                block += "\n" + f"Tools:\n{tools_desc}"
            else:
                block += "\n  - No tools available."
            blocks.append(block)

        full_desc = "\n\n".join(blocks)
        descriptions.append(full_desc)

    return descriptions

completed_test_case_names = []

def load_completed_tasks(output_dir: str):
    """
    加载 output_dir 下所有 .json 文件，提取 (npx_name, server_version) 作为已完成任务集合。

    假设每个 JSON 文件包含字段:
      - "npx_name"
      - "server_version"

    返回: set of (npx_name, server_version)
    """
    global completed_test_case_names
    if not os.path.exists(output_dir):
        return
    completed_test_case_names = list(os.listdir(output_dir))

def run_all_with_progress(
    task_prompts: List[Tuple[str, str, int]],
    output_dir: str = "outputs",
    model_list:str = ["deepseek-chat"],
):
    tasks = []
    for prompt, category, idx in task_prompts:
        for model_name in model_list:
            tasks.append((prompt, model_name, category, idx, output_dir))

    def _run_task(args):
        return generate_save_with_one_server(*args)

    success_count = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run_task, task_args) for task_args in tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                success_count += 1
    print(f"\n✅ 成功完成 {success_count} / {len(tasks)} 个任务")

def generate_save_with_one_server(
        prompt: str,
        model_name: str,
        category: str,
        idx: int,
        output_dir: str
):
    try:
        safe_file_name = f"{category}@@{idx}@@{model_name}.json".replace("/","_").replace( "\\", "_")
        if safe_file_name in completed_test_case_names:
            return True
        response = chat_with_openai(
            conversation=[{'role':'user', 'content': prompt}],
            model=model_name,
            temperature=1.0,
            max_tokens=4096
        )
        if response is None:
            print(1)
        xml_test_case = parse_mutil_mcp_tool_use_xml(response)
        filepath = os.path.join(output_dir, safe_file_name)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "response": response,
                "question": xml_test_case['question'],
                "tool_calls": xml_test_case['tool_calls'],
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"\n⚠️  Error in calling model {model_name}: {e}")
        return False

    return True

def load_multiple_json_lists(folder_path):
    """
    从多个 JSON 文件中加载列表，并合并为一个列表。

    参数:
        json_paths (list): JSON 文件路径列表，如 ['a.json', 'b.json']

    返回:
        list: 所有 JSON 文件中列表内容的合并结果
    """

    def parse_xml_string(xml_string: str) -> Dict[str, List[str]]:
        """
        从XML文件解析server和tool信息
        """
        try:
            root = ET.fromstring(xml_string.strip())
        except ET.ParseError as e:
            # print(f"Invalid XML: {e}")
            return {}
        server_tools = {}
        target_servers = root.find('target_servers')
        if target_servers is not None:
            for server in target_servers.findall('server'):
                server_name = server.get('name')
                tools = [tool.text.strip() for tool in server.findall('tool') if tool.text]
                server_tools[server_name] = tools
        return server_tools

    # 使用示例
    # server_tools = parse_xml_file('response.xml')

    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    combined_list = []

    for file_name in file_list:
        try:
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                xml_test_case = extract_xml_from_markdown(data['response'].strip())
                server_and_tools = parse_xml_string(xml_test_case)
                data['servers'] = server_and_tools
            combined_list.append(data)  # 合并列表
        except FileNotFoundError:
            print(f"❌ 错误: 文件未找到 - {file_name}")
        except json.JSONDecodeError as e:
            print(f"❌ 错误: JSON 格式无效 - {file_name} ({e})")
        except Exception as e:
            print(f"❌ 未知错误 - {file_name}: {e}")
    return combined_list

def reverse_mapping(server_tool_dict: dict) -> dict:
    """
    将 {server: [tools]} 映射反向为 {tool: server}

    Args:
        server_tool_dict: 原始字典，key为server，value为tool列表

    Returns:
        反向字典，key为tool，value为server
    """
    tool_to_server = {}

    for server, tools in server_tool_dict.items():
        for tool in tools:
            tool_to_server[tool] = server

    return tool_to_server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Classify the functions of the servers and randomly combine them; "
                    "Step 2: Generate a problem from the server combination; "
                    "Step 3: Use Server based tools to complete task generation trajectories; "
                    "Step 4: Use mutation based tools to complete task generation trajectories;")
    parser.add_argument(
        "--step", default='1', choices=['1','2','3', '4']
    )
    parser.add_argument(
        "--output_path", default='custom/data/test_cases/',
    )
    parser.add_argument(
        "--model_name", default='gpt-4o'
    )
    parser.add_argument(
        "--combination_num", default=100, type=int
    )
    parser.add_argument(
        "--part_num", default=-1, type=int
    )
    parser.add_argument(
        "--part_idx", default=0, type=int
    )
    parser.add_argument(
        "--mutation_type", default='', choices=['HYBRID']
    )
    parser.add_argument(
        "--mutation_path", default='custom/data/mutation/HYBRID', type=str
    )
    parser.add_argument(
        "--test_case_file", default='custom/data/test_cases/test_cases.json', type=str
    )
    parser.add_argument(
        "--mcp_server_url", default='http://127.0.0.1:3050/mcp', type=str
    )
    #  python custom/real_mcp/tool_level_test.py --test_case_file custom/data/test_cases/test_cases.json --model_name gpt-5.4 --step 4 --mcp_server_url http://127.0.0.1:4060/mcp --output_path custom/data/exp_output/ --mutation_path custom/data/mutation/HYBRID/HYBRID --part_num 4 --part_idx 1
    parser.add_argument(
        "--fschat_flag",
        action='store_true',
        help="Enable FastChat format conversion and inference mode"
    )
    args = parser.parse_args()
    # assert args.mutation_type != ''

    # output_number = 0
    tool_status, tool_names = asyncio.run(get_1mcp_status(args.mcp_server_url))

    server_name_set = set()
    for tool_name in tool_names:
        mcp_server_name = tool_name.split('_1mcp_')[0]
        server_name_set.add(mcp_server_name)

    print('Server Number:', len(server_name_set))
    with open('custom/data/test_cases/server_description_map.json', 'r', encoding='utf-8') as file:
        server_description_map = json.load(file)

    RESULT_OUTPUT = args.output_path + f'/{args.model_name}/'
    os.makedirs(RESULT_OUTPUT, exist_ok=True)
    if args.step == '1':
        print(f"\n{'=' * 60}")
        print(f"🚀 First step for question generation, category mcp server in custom/data/test_cases/server_category.json")
        print(f"{'=' * 60}\n")

        classification_prompt = build_mcp_classification_prompt(server_description_map)
        response = chat_with_openai(
            conversation=[{'role':'user', 'content': classification_prompt}],
            model=args.model_name,
            temperature=1.0,
            max_tokens=8192
        )
        classification_json = parse_string_to_json(response)
        print(len(classification_json["assignments"]))

        with open('custom/data/test_cases/server_category.json', 'w', encoding='utf-8') as f:
            json.dump(classification_json, f, ensure_ascii=False, indent=2)
    elif args.step == '2':
        task_prompts = []
        with open('custom/data/test_cases/server_category.json', 'r', encoding='utf-8') as f:
            server_category_status = json.load(f)
        # categories = [item['name'] for item in server_category_status['categories']]
        category_to_servers = defaultdict(list)
        for server_name, categories in server_category_status['assignments'].items():
            for cat in categories:
                category_to_servers[cat].append(server_name)

        task_prompts = []
        category_combination_prompts = {}
        for category, servers in category_to_servers.items():
            server_combinations = generate_combinations(servers, num_combinations=args.combination_num)
            server_descriptions = generate_multi_tool_descriptions(server_combinations, server_description_map, tool_status)
            for idx, server_desc in enumerate(server_descriptions):
                servers = server_combinations[idx]
                query_prompt = MULTI_SERVER_MULTI_TOOL_PROMPT.format(
                        SERVER_DESCRIPTIONS=server_desc,
                        FILE_EXISTS=f'The current path is {os.getcwd()}, and the directory below is: \n' + generate_tree_string('./anotation_path'),
                )
                if category not in category_combination_prompts.keys():
                    category_combination_prompts[category] = []
                category_combination_prompts[category].append({
                    'idx': idx,
                    'servers': servers,
                    'tool_description': server_desc,
                    'task_prompt': query_prompt
                })
                task_prompts.append((query_prompt, category, idx))

        os.makedirs('custom/data/test_cases/TASK', exist_ok=True)
        load_completed_tasks('custom/data/test_cases/TASK')
        # run_all_with_progress(task_prompts, RESULT_OUTPUT, [args.model_name])
        for _prompt in tqdm(task_prompts):
            generate_save_with_one_server(_prompt[0], args.model_name, _prompt[1], _prompt[2], 'custom/data/test_cases/TASK')
    elif args.step == '3':
        involved_tool_numbers = []
        with open(args.test_case_file, 'r', encoding='utf-8') as f:
            all_test_cases = json.load(f)
        all_test_cases = clean_all_test_cases(all_test_cases)
        if args.part_num == -1:
            bias_number = 0
            pass
        else:
            assert args.part_idx < args.part_num
            total_number = len(all_test_cases)
            bias_number = (total_number * args.part_idx)//args.part_num
            all_test_cases = all_test_cases[bias_number: bias_number + total_number//args.part_num]
        os.makedirs(RESULT_OUTPUT+'/ANSWER/', exist_ok=True)
        existing_ids = os.listdir(RESULT_OUTPUT+'/ANSWER/')
        for testcase_idx, test_case in tqdm(enumerate(all_test_cases), total=len(all_test_cases), desc="测试用例进度", unit="case", colour='blue'):
            if 'idx' in test_case:
                real_idx = test_case['idx']
            else:
                raise KeyError
            involved_servers = test_case['servers']
            involved_tools = return_involved_tool_status(involved_servers, tool_status)
            # if len(involved_tools) > 8:
            #     continue

            involved_tool_numbers.append(len(involved_tools))
            if involved_tools == []:
                print('involved_tools is empty')
                continue

            openai_agent = ChatModel(
                model_name=args.model_name,
                api_key='your api key',
                model_url='your api base url',
                tools=involved_tools,
                max_round=10 if 2*len(involved_tools) < 10 else 2*len(involved_tools),
                tool_server_map=reverse_mapping(involved_servers),
                file_exists=generate_tree_string('./anotation_path'),
                mcp_server_url=args.mcp_server_url
            )
            answer, trajectory_messages, stop_flag, exceeded_flag = openai_agent.run_with_server(test_case['question'], args.fschat_flag)
            # if not exist_tool_calls(trajectory_messages):
            #     continue
            test_case = clean_test_case(test_case)
            test_case['idx'] = real_idx
            test_case['answer'] = answer
            test_case['trajectory_messages'] = trajectory_messages
            test_case['stop'] = stop_flag
            test_case['exceeded'] = exceeded_flag
            test_case['involved_tools'] = involved_tools

            with open(RESULT_OUTPUT + f'/ANSWER/{real_idx}.json', "w", encoding="utf-8") as f:
                json.dump(test_case, f, ensure_ascii=False, indent=2)
    elif args.step == '4': # --mutation_path custom/data/mutation/HYBRID
        RESULT_OUTPUT = args.mutation_path
        os.makedirs(RESULT_OUTPUT+f'/ANSWER-{args.model_name}/', exist_ok=True)
        existing_ids = os.listdir(RESULT_OUTPUT+f'/ANSWER-{args.model_name}/')
        involved_tool_numbers = []
        with open(args.test_case_file, 'r', encoding='utf-8') as f:
            all_test_cases = json.load(f)
        all_test_cases = clean_all_test_cases(all_test_cases)

        all_test_cases = return_involved_mutated_test_cases(all_test_cases, args)
        if args.part_num == -1:
            bias_number = 0
            pass
        else:
            assert args.part_idx < args.part_num
            total_number = len(all_test_cases)
            bias_number = (total_number * args.part_idx)//args.part_num
            all_test_cases = all_test_cases[bias_number: bias_number + total_number//args.part_num]

        for _, test_case in tqdm(enumerate(all_test_cases), total=len(all_test_cases), desc="测试用例进度", unit="case", colour='blue'):
            origin_tool_number = len(test_case['tool_calls'])
            test_case = change_test_case_servers(test_case, args) # 继承了上一个演化
            real_idx = test_case['idx']

            if f'{real_idx}.json' in existing_ids:
                continue
            involved_tools = return_involved_mutated_tool_status(test_case['servers'], tool_status, args)
            # if len(involved_tools) < len(test_case['tool_calls']):
            #     print(real_idx, len(test_case['tool_calls'])-len(involved_tools))
            assert involved_tools is not None
            #
            involved_tool_numbers.append(len(involved_tools))
            if involved_tools == []:
                print('involved_tools is empty')
                continue
            openai_agent = ChatModel(
                model_name=args.model_name,
                api_key='your api key',
                model_url='your api base url',
                tools=involved_tools,
                max_round=10 if 2*origin_tool_number<10 else 2*origin_tool_number,
                tool_server_map=reverse_mapping(test_case['servers']),
                file_exists=generate_tree_string('./anotation_path'),
                mcp_server_url=args.mcp_server_url
            )
            # print(test_case['question'])
            answer, trajectory_messages, stop_flag, exceeded_flag = openai_agent.run_with_server(test_case['question'], args.fschat_flag)
            # if not exist_tool_calls(trajectory_messages):
            #     continue
            test_case['answer'] = answer
            test_case['trajectory_messages'] = trajectory_messages
            test_case['stop'] = stop_flag
            test_case['exceeded'] = exceeded_flag
            test_case['involved_tools'] = involved_tools

            with open(RESULT_OUTPUT + f'/ANSWER-{args.model_name}/{real_idx}.json', "w", encoding="utf-8") as f:
                json.dump(test_case, f, ensure_ascii=False, indent=2)