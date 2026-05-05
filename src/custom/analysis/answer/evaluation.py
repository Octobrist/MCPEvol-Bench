"""
基于 eval_error_prompt.build_mcp_bench_judge_prompt 对单条轨迹做 LLM 评分。
默认对每条 case 评估 5 次，每次随机打乱三个维度的 rubric 与 Step 1/2 条目顺序（prompt shuffle）。
测试数据为列表，元素含 question, trajectory_messages, answer, servers。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from custom.prompt.eval_error_prompt import build_mcp_bench_judge_prompt
from custom.utils.test_utils import load_all_json_files, format_tools_to_string, get_1mcp_server_tools
from custom.utils.mcp_utils import get_1mcp_status
from custom.utils.openai_llm import chat_with_openai
from custom.analysis.mutation.utils import extract_json_from_markdown

tool_status, _ = asyncio.run(get_1mcp_status('http://127.0.0.1:3050/mcp'))

NUM_EVAL_RUNS = 5
# 默认同时评估的任务数（每条 case 视为一个 task）；受 API 限流时可调小
DEFAULT_MAX_WORKERS = 5
# 维度下标：0=Task Fulfillment, 1=Information Grounding, 2=Planning Efficiency


def messages_to_string(messages: list[dict]) -> str:
    """
    将 OpenAI 风格的 messages 列表转换为简要字符串。
    - user/assistant 文本直接保留
    - assistant 的 tool_calls：每条含 tool_call_id（对应 tc 的 id），便于与 tool 消息中的 tool_call_id 配对
    - tool 响应带 tool_call_id，可与上述调用行配对
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        # 1. 处理普通文本内容
        # if content:
        #     parts.append(f"[{role}]: {content}")
        # 2. 处理工具调用 (通常在 assistant 消息中)
        if tool_calls := msg.get("tool_calls"):
            for tc in tool_calls:
                tool_id = tc.get("id", None)
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                # 尝试格式化 JSON 以便阅读，如果失败则保留原样
                try:
                    args_str = json.dumps(json.loads(args), ensure_ascii=False, separators=(',', ':'))
                except Exception:
                    args_str = args
                parts.append(
                    f"[{role} -> Tool]: tool_call_id={tool_id!r} Call {name}({args_str})"
                )
        # 3. 处理工具执行结果 (可选，通常内容较短)
        if role == "tool" and content:
            tcid = msg.get("tool_call_id", None)
            parts.append(f"[Tool Result tool_call_id={tcid!r}]: {content}")  # 限制长度防止过长
        else:
            parts.append(f"[{role}]: {content}")
    return "\n".join(parts)

def get_tool_descriptions(server_tool_dict):
    full_desc = ''
    for server_name, tool_names in server_tool_dict.items():
        full_desc += f'Tools for Server - {server_name}: \n'
        tools = get_1mcp_server_tools(tool_status, server_name)
        tools = [tool for tool in tools if tool.name in tool_names]
        if tools:
            tools_desc = format_tools_to_string(tools, dict_flag=False)
            full_desc += f'{tools_desc}\n'
        full_desc += '\n'
    return full_desc

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

def build_judge_user_content(
    case: Dict[str, Any],
    *,
    dimension_order: Tuple[int, int, int] = (0, 1, 2),
) -> str:
    """将单条 case 填入评判 prompt（用 replace：prompt 末尾含 JSON 示例，不能 str.format）。"""
    task = case.get("question", "")
    answer = case.get("answer", "")
    available_tools = get_tool_descriptions(case['servers'])
    trajectory = truncate_tool_content(case.get("trajectory_messages"))[2:]
    trajectory_string = messages_to_string(trajectory)
    return build_mcp_bench_judge_prompt(
        task,
        available_tools,
        trajectory_string,
        answer,
        dimension_order=dimension_order,
    )

def evaluate_one(
    case: Dict[str, Any],
    *,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 4096*2,
    dimension_order: Tuple[int, int, int] = (0, 1, 2),
) -> Dict[str, Any]:
    """
    对一条 case 调用评判模型，返回解析后的 JSON（含三个维度分数与 reasoning）。
    失败时返回 {"error": "...", "raw": "..."}。
    """
    user_content = build_judge_user_content(case, dimension_order=dimension_order)
    raw = chat_with_openai(
        conversation=[{"role": "user", "content": user_content}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if raw is None:
        return {"error": "LLM call failed or returned empty", "raw": None}

    parsed = extract_json_from_markdown(raw.strip())
    if parsed is None:
        return {"error": "Failed to parse judge JSON", "raw": raw}
    return parsed


def _random_dimension_order(rng: random.Random) -> Tuple[int, int, int]:
    perm = [0, 1, 2]
    rng.shuffle(perm)
    return (perm[0], perm[1], perm[2])


def _mean_int(vals: List[int]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def aggregate_eval_runs(run_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对多次成功解析的分数取算术平均；reasoning 保留各次列表。"""
    successful: List[Dict[str, Any]] = []
    for item in run_payloads:
        r = item.get("result")
        if isinstance(r, dict) and "error" not in r:
            successful.append(r)
    tf = [int(x["task_fulfillment"]) for x in successful if isinstance(x.get("task_fulfillment"), int)]
    gr = [int(x["grounding"]) for x in successful if isinstance(x.get("grounding"), int)]
    pe = [int(x["planning_and_efficiency"]) for x in successful if isinstance(x.get("planning_and_efficiency"), int)]
    tf_m, gr_m, pe_m = _mean_int(tf), _mean_int(gr), _mean_int(pe)
    out: Dict[str, Any] = {
        "successful_runs": len(successful),
        "total_runs": len(run_payloads),
        "task_fulfillment_mean": tf_m,
        "grounding_mean": gr_m,
        "planning_and_efficiency_mean": pe_m,
        "task_fulfillment_reasoning_runs": [x.get("task_fulfillment_reasoning") for x in successful],
        "grounding_reasoning_runs": [x.get("grounding_reasoning") for x in successful],
        "planning_efficiency_reasoning_runs": [x.get("planning_efficiency_reasoning") for x in successful],
    }
    # 与单次评估相同的整数字段名，取成功运行均值的四舍五入，便于下游沿用
    if tf_m is not None:
        out["task_fulfillment"] = int(round(tf_m))
    if gr_m is not None:
        out["grounding"] = int(round(gr_m))
    if pe_m is not None:
        out["planning_and_efficiency"] = int(round(pe_m))
    return out


def evaluate_one_five_times(
    case: Dict[str, Any],
    *,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 4096 * 2,
    num_runs: int = NUM_EVAL_RUNS,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    对单条 case 评估 num_runs 次；每次随机打乱三个维度在 rubric 与 Step 1/2 中的顺序。
    返回 {"runs": [...], "aggregate": {...}}，aggregate 对成功解析的分数取均值。
    """
    rng = rng or random.Random()
    runs: List[Dict[str, Any]] = []
    # NOTE: 评估过程中可能出现 LLM 返回不可解析 JSON，导致 successful_runs < num_runs。
    # 这里按“成功解析的次数”补跑，直到成功次数达到 num_runs（默认 5）。
    # 为避免一直失败造成死循环，设置最大总尝试次数上限。
    target_successful_runs = max(1, int(num_runs))
    max_total_runs = max(target_successful_runs, target_successful_runs * 3)
    while True:
        successful_so_far = sum(
            1
            for item in runs
            if isinstance(item.get("result"), dict) and "error" not in item.get("result", {})
        )
        if successful_so_far >= target_successful_runs:
            break
        if len(runs) >= max_total_runs:
            break
        order = _random_dimension_order(rng)
        result = evaluate_one(
            case,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            dimension_order=order,
        )
        runs.append({"dimension_order": list(order), "result": result})
    aggregate = aggregate_eval_runs(runs)
    packed: Dict[str, Any] = {"runs": runs, "aggregate": aggregate}
    # 将聚合后的整数分数字段也挂在顶层，兼容原先 eval_result 为扁平 JSON 的用法
    for k in ("task_fulfillment", "grounding", "planning_and_efficiency"):
        if k in aggregate:
            packed[k] = aggregate[k]
    return packed


def count_mutation_scores(
    cases,
    *,
    save_dir: str,
    mutation_save_dir: str,
):
    """从已有 EVAL 结果汇总 mutation 分数（顺序读取，无 LLM）。"""
    total_scores = {'task_fulfillment_mean':0, 'grounding_mean':0, 'planning_and_efficiency_mean':0}
    total_number = 0
    mutation_number = 0
    for _, case in tqdm(cases.items()):
        if os.path.exists(f'{mutation_save_dir}/{case["idx"]}.json'):
            with open(f'{mutation_save_dir}/{case["idx"]}.json', 'r', encoding='utf-8') as f:
                eval_case = json.load(f)
                mutation_number += 1
        else:
            # raise KeyError
            with open(f'{save_dir}/{case["idx"]}.json', 'r', encoding='utf-8') as f:
                eval_case = json.load(f)
        packed = eval_case['eval_result']
        total_scores['task_fulfillment_mean'] += packed['aggregate']['task_fulfillment_mean']
        total_scores['grounding_mean'] += packed['aggregate']['grounding_mean']
        total_scores['planning_and_efficiency_mean'] += packed['aggregate']['planning_and_efficiency_mean']
        total_number += 1

    print(f"task_fulfillment: {total_scores['task_fulfillment_mean'] / total_number:.3f}")
    print(f"grounding: {total_scores['grounding_mean'] / total_number:.3f}")
    print(f"planning_and_efficiency: {total_scores['planning_and_efficiency_mean'] / total_number:.3f}")
    print(mutation_number, total_number)

def _evaluate_case_task(
    case: Dict[str, Any],
    *,
    model: str,
    num_runs: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """供线程池调用的单 task 评估（独立 RNG，避免多线程共享 random 状态）。"""
    rng = random.Random(rng_seed)
    return evaluate_one_five_times(case, model=model, num_runs=num_runs, rng=rng)


def evaluate_list(
    cases,
    *,
    model: str = "gpt-4o",
    save_dir: Optional[str] = None,
    num_runs: int = NUM_EVAL_RUNS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    random_seed: int = 0,
):
    """评估每条轨迹；对多个 task 使用线程池并发（每条 case 内仍为顺序多次 LLM 调用）。

    若提供 save_dir，则每评估一条即写入 save_dir/{idx}.json（idx 为从 0 起的序号）。
    每条 case 默认评估 num_runs 次（prompt shuffle），结果写入 case['eval_result']。
    """
    total_scores = {'task_fulfillment_mean':0, 'grounding_mean':0, 'planning_and_efficiency_mean':0}
    total_number = 0
    assert save_dir is not None

    def _is_eval_complete(path: str, *, required_successful_runs: int) -> bool:
        """判断已落盘的 EVAL 文件是否满足所需评估次数（默认 5）。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False

        eval_result = payload.get("eval_result")
        if not isinstance(eval_result, dict):
            return False

        aggregate = eval_result.get("aggregate")
        if not isinstance(aggregate, dict):
            return False

        successful_runs = aggregate.get("successful_runs")
        if not isinstance(successful_runs, int):
            return False

        return successful_runs >= int(required_successful_runs)

    items = list(cases.items())
    need_eval: List[Tuple[Any, Dict[str, Any]]] = []
    for key, case in items:
        path = os.path.join(save_dir, f"{case['idx']}.json")
        # 即使文件存在，也要检查是否满足 required_successful_runs（默认 5）
        if (not os.path.exists(path)) or (not _is_eval_complete(path, required_successful_runs=num_runs)):
            need_eval.append((key, case))

    workers = max(1, min(max_workers, len(need_eval))) if need_eval else 1
    # need_eval = False
    if need_eval:
        def _run_one(key_case: Tuple[Any, Dict[str, Any]]) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
            key, case = key_case
            idx = int(case.get("idx", 0))
            rng_seed = (random_seed * 1_000_003) ^ (idx * 1_000_033)
            packed = _evaluate_case_task(case, model=model, num_runs=num_runs, rng_seed=rng_seed)
            return key, case, packed

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_one, kc): kc[0] for kc in need_eval}
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="tasks (concurrent)",
            ):
                key, case, packed = fut.result()
                case["eval_result"] = packed
                with open(os.path.join(save_dir, f"{case['idx']}.json"), "w", encoding="utf-8") as f:
                    json.dump(case, f, indent=2, ensure_ascii=False)

    for key, case in tqdm(items, desc="aggregate", leave=False):
        path = os.path.join(save_dir, f"{case['idx']}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            packed = loaded.get("eval_result", {})
            if not isinstance(packed, dict) or "aggregate" not in packed:
                continue

        total_scores["task_fulfillment_mean"] += packed["aggregate"]["task_fulfillment_mean"]
        total_scores["grounding_mean"] += packed["aggregate"]["grounding_mean"]
        total_scores["planning_and_efficiency_mean"] += packed["aggregate"]["planning_and_efficiency_mean"]
        total_number += 1
    print(f"task_fulfillment: {total_scores['task_fulfillment_mean'] / total_number:.3f}")
    print(f"grounding: {total_scores['grounding_mean'] / total_number:.3f}")
    print(f"planning_and_efficiency: {total_scores['planning_and_efficiency_mean'] / total_number:.3f}")
    print(total_number)

# --answer_path custom/data/exp_output/gpt-5.1/ANSWER --mutation_answer_path custom/data/mutation/HYBRID/ANSWER-gpt-5.1
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trajectories")
    parser.add_argument(
        "--answer_path", default='custom/data/exp_output/deepseek-chat/ANSWER', help="Path to JSON list of cases (question, trajectory_messages, answer, servers)",
    )
    parser.add_argument(
        "--mutation_answer_path", default='', help="Path to JSON list of cases (question, trajectory_messages, answer, servers)",
    )
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--num_runs", type=int, default=NUM_EVAL_RUNS, help="Evaluations per case (prompt shuffle each run)")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Concurrent tasks (cases) when calling the judge API; lower if rate-limited",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Base seed for per-task RNG (shuffle order); reproducible across runs",
    )
    args = parser.parse_args()
    print(f'will be evaluated: {args.answer_path}')

    answer_path = args.answer_path
    assert 'ANSWER' in answer_path
    save_dir = answer_path.replace('ANSWER', 'EVAL')
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(answer_path):
        print(f"Answer file {answer_path} does not exist")
        exit(1)

    cases = load_all_json_files(answer_path)

    with open(f'custom/data/test_cases/real.json', 'r', encoding='utf-8') as f:
        real_test_cases = json.load(f)
    real_test_case_ids = [item['idx'] for item in real_test_cases['samples']]

    print(f'model name: {args.answer_path}, real test cases: {len(real_test_case_ids)}')
    cases = {
        key: case
        for key, case in cases.items()
        if case['idx'] in real_test_case_ids
    }
    print(f"fliter case number: {len(cases)}")
    if args.mutation_answer_path != '':
        mutation_answer_path = args.mutation_answer_path
        mutation_save_dir = mutation_answer_path.replace('ANSWER', 'EVAL')
        count_mutation_scores(cases, save_dir=save_dir, mutation_save_dir=mutation_save_dir)
    else:
        evaluate_list(
            cases,
            model=args.model,
            save_dir=save_dir,
            num_runs=args.num_runs,
            max_workers=args.max_workers,
            random_seed=args.random_seed,
        )
    print(f'final evaluation: {args.answer_path}')