import base64
import os

import pytz

import mcp
from mcp.client.streamable_http import streamablehttp_client
import httpx  # 假设底层使用 httpx；如果不是，请替换为对应异常类型
import json
from datetime import datetime
from pathlib import Path

import asyncio
from typing import Any, Dict

smithery_api_pool = None


def check_if_api_key_is_valid(profile, smithery_api_key):
    """
    Check if a profile and Smithery API key combination is valid by making a test request to Smithery API.
    Uses a simple HTTP request approach to avoid async complications.

    Args:
        profile (str): The profile name to test
        smithery_api_key (str): The Smithery API key to test

    Returns:
        dict: {"valid": bool, "message": str, "tools": list or None}
    """
    try:
        # Create configurations
        config = {"debug": False}
        config_b64 = base64.b64encode(json.dumps(config).encode()).decode()

        # Create server URL - testing the Smithery API key from the pool
        url = f"https://server.smithery.ai/exa/mcp?config={config_b64}&api_key={smithery_api_key}&profile={profile}"
        # Simple HTTP test to check if the endpoint is accessible
        import requests
        try:
            # headers = {"Authorization": "Bearer b2dd2aaa-fc00-4070-b2c5-1881e959096e"}
            # Test with a simple GET request first to see if the endpoint responds
            response = requests.get(url, timeout=10)

            # If we get any response (even an error), it means the API key format is likely correct
            # and the service is accessible. For more detailed validation, we'd need MCP.
            if response.status_code == 200:
                return {
                    "valid": True,
                    "message": "Successfully connected via HTTP. Endpoint accessible.",
                    "tools": ["http_validated"]  # Simple indicator
                }
            elif response.status_code == 401:
                return {
                    "valid": False,
                    "message": "Authentication failed. Invalid API key or profile.",
                    "tools": None
                }
            elif response.status_code == 403:
                return {
                    "valid": False,
                    "message": "Access forbidden. Check API key permissions.",
                    "tools": None
                }
            elif response.status_code == 404:
                return {
                    "valid": False,
                    "message": "Endpoint not found. Check profile name.",
                    "tools": None
                }
            else:
                return {
                    "valid": True,
                    "message": f"Endpoint accessible (HTTP {response.status_code}).",
                    "tools": ["http_validated"]
                }

        except requests.exceptions.Timeout:
            return {
                "valid": False,
                "message": "Connection timeout. Service may be down.",
                "tools": None
            }
        except requests.exceptions.ConnectionError:
            return {
                "valid": False,
                "message": "Connection failed. Network or service issue.",
                "tools": None
            }
        except requests.exceptions.RequestException as e:
            return {
                "valid": False,
                "message": f"HTTP request failed: {str(e)}",
                "tools": None
            }

    except Exception as e:
        return {
            "valid": False,
            "message": f"Error during API validation: {str(e)}",
            "tools": None
        }


def validate_api_pool_entry(entry):
    """
    Validate a single entry from the API pool.

    Args:
        entry (dict): Entry with 'profile' and 'api_key' fields

    Returns:
        dict: Validation result with additional entry info
    """
    if not isinstance(entry, dict) or 'profile' not in entry or 'api_key' not in entry:
        return {
            "valid": False,
            "message": "Invalid entry format. Must have 'profile' and 'api_key' fields.",
            "profile": entry.get('profile', 'unknown'),
            "source": entry.get('source', 'unknown'),
            "tools": None
        }

    # Test the Smithery API key from the entry
    result = check_if_api_key_is_valid(entry['profile'], entry['api_key'])

    # Add entry metadata to result
    result['profile'] = entry['profile']
    result['source'] = entry.get('source', 'unknown')

    return result

def validate_api_pool_from_file(file_path):
    """
    Validate all entries in an API pool JSON file.

    Args:
        file_path (str): Path to the JSON file containing API pool

    Returns:
        dict: Summary of validation results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if 'api_pool' not in data:
            return {
                "error": "File must contain 'api_pool' key with list of entries",
                "results": []
            }

        results = []
        valid_count = 0

        for i, entry in enumerate(data['api_pool']):
            print(f"Validating entry {i + 1}/{len(data['api_pool'])}: {entry.get('profile', 'unknown')}")

            result = validate_api_pool_entry(entry)
            results.append(result)

            if result['valid']:
                valid_count += 1
                print(f"  ✓ Valid - {result['message']}")
            else:
                print(f"  ✗ Invalid - {result['message']}")

        return {
            "total_entries": len(data['api_pool']),
            "valid_entries": valid_count,
            "invalid_entries": len(data['api_pool']) - valid_count,
            "results": results
        }

    except Exception as e:
        return {
            "error": f"Failed to process file: {str(e)}",
            "results": []
        }

def load_and_validate_smithery_api_pool(pool_file_path):
    """Load and validate Smithery API pool from JSON file, keeping only valid keys"""
    global smithery_api_pool

    print("=" * 50)
    print("🔍 SMITHERY API POOL VALIDATION")
    print("=" * 50)

    # Check if pool file exists
    if not os.path.exists(pool_file_path):
        print(f"⚠️  API pool file {pool_file_path} not found!")
        print("🔍 Testing fallback API key from arguments...")
        return None

    # Validate the entire API pool using the test logic
    print(f"📁 Validating all entries in {pool_file_path}...")

    try:
        results = validate_api_pool_from_file(pool_file_path)

        if "error" in results:
            print(f"❌ Error: {results['error']}")
            raise ValueError(f"API pool validation failed: {results['error']}")

        # Display detailed results like in test file
        print("=" * 30)
        print("📊 VALIDATION SUMMARY")
        print("=" * 30)
        print(f"Total entries: {results['total_entries']}")
        print(f"Valid entries: {results['valid_entries']}")
        print(f"Invalid entries: {results['invalid_entries']}")
        print(f"Success rate: {results['valid_entries'] / results['total_entries'] * 100:.1f}%")

        print(f"\n📋 DETAILED RESULTS")
        print("-" * 30)
        for result in results['results']:
            status = "✅" if result['valid'] else "❌"
            print(f"{status} {result['profile']} ({result['source']}): {result['message']}")

        # Check if we have any valid entries
        if results['valid_entries'] == 0:
            raise ValueError("❌ No valid API keys found in the pool! All API keys failed validation.")

        # Load original data to get valid entries with API keys
        with open(pool_file_path, 'r') as f:
            original_data = json.load(f)
            original_pool = original_data.get('api_pool', [])

        # Keep only valid entries
        valid_pool = []
        for result in results['results']:
            if result['valid']:
                # Find the original entry to get the API key
                for original_entry in original_pool:
                    if original_entry['profile'] == result['profile']:
                        valid_pool.append(original_entry)
                        break

        smithery_api_pool = valid_pool

        print(f"\n✅ SUCCESS: Using {len(smithery_api_pool)} valid API keys from pool")
        print("=" * 50)
        return smithery_api_pool

    except Exception as e:
        print(f"❌ Error during API pool validation: {e}")
        raise ValueError(f"API pool validation failed: {str(e)}")

def create_date_dir(base_output_dir="output"):
    beijing_tz = pytz.timezone("Asia/Shanghai")
    minute_str = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M")
    output_path = Path(base_output_dir) / minute_str
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def construct_mcp_server_url(server_url, api_key=None, profile=None):
    """
    Construct MCP server URL from server info.
    """
    return f"{server_url}?api_key={api_key}"
    # return f"{server_url}?api_key={api_key}&profile={profile}"

import sys
from typing import List

def extract_all_exceptions(exc: BaseException) -> List[BaseException]:
    """递归提取 ExceptionGroup / BaseExceptionGroup 中的所有叶子异常"""
    if sys.version_info >= (3, 11) and isinstance(exc, BaseExceptionGroup):
        leaves = []
        for sub_exc in exc.exceptions:
            leaves.extend(extract_all_exceptions(sub_exc))
        return leaves
    else:
        return [exc]

async def get_mcp_tools(mcp_server_url):
    try:
        # async with mcp_server_context as mcp_server:
        #     tools_result = await mcp_server.list_tools()
        #     return tools_result
        async with streamablehttp_client(mcp_server_url) as (read_stream, write_stream, _):
            async with mcp.ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                return tools_result
    except asyncio.exceptions.CancelledError as aec:
        raise Exception(f"asyncio error: {str(aec)}")
    except httpx.HTTPStatusError as httperr:
        raise Exception(f"http error: {str(httperr)}")
    except BaseExceptionGroup as eg:
        # 👈 关键：先于 Exception 捕获 BaseExceptionGroup！
        leaf_exceptions = extract_all_exceptions(eg)
        # 取第一个子异常（通常只有一个），或拼接多个
        first = leaf_exceptions[0]
        # 如果是 HTTP 401，我们想特别提示
        if isinstance(first, httpx.HTTPStatusError) and first.response.status_code == 401:
            raise Exception(f"HTTP 401 Unauthorized")
        else:
            # 通用处理：把所有子异常转成字符串
            messages = [f"{type(e).__name__}: {e}" for e in leaf_exceptions]
            raise Exception("MCP server error(s): " + " | ".join(messages)) from eg
    except Exception as e:
        raise Exception(f"Unexpected error in MCP communication: {type(e).__name__}: {str(e)}") from e

def format_tool(tool):
    new_tool = {}
    new_tool['name'] = tool.name
    new_tool['description'] = tool.description
    new_tool['inputSchema'] = tool.inputSchema
    new_tool['outputSchema'] = tool.outputSchema
    return new_tool


from pathlib import Path

async def fetch_all_tools_sequential(
        file_server_mapping: Dict[str, Any],
        delay_per_request: float = 2.0,
        show_progress: bool = True,
        output_dir: str = "output",
) -> Dict[str, Any]:
    """
    顺序（串行）获取所有 MCP 服务的工具列表，每次请求后可选等待一段时间。

    Args:
        file_server_mapping: dict，key=文件名（str），value=mcp_server_context
        delay_per_request: 每次请求完成后等待的秒数（默认 0.0，即不等待）
        show_progress: 是否显示进度（简单打印当前处理的文件名）

    Returns:
        dict: {filename: tools_result 或 error message}
    """
    results = {}
    total = len(file_server_mapping)

    for idx, (filename, context) in enumerate(file_server_mapping.items(), 1):
        if show_progress:
            print(f"[{idx}/{total}] Fetching tools from {filename}...")

        save_name = filename.replace("/", "_")
        if save_name == 'vercel-domains' or save_name == '@DynamicEndpoints_espn-mcp' or save_name == '@nate_kismet-travel' or save_name == '@rycurc_glystnmcp' or save_name == '@aryankeluskar_canvas-mcp':
            continue
        save_file = os.path.join(output_dir, f"smithery-{save_name}.json")
        file_path = Path(save_file)
        if file_path.exists():
            continue

        try:
            list_results = await get_mcp_tools(context)
            results[filename] = list_results
        except Exception as e:
            results[filename] = f"Error when fetching tools: {str(e)}"

        # 可选：每次请求后暂停
        if delay_per_request > 0:
            await asyncio.sleep(delay_per_request)

        if not isinstance(results[filename], str):
            print('✅, find one')
            tools = list_results.tools if hasattr(list_results, "tools") else []
            tools = [format_tool(tool) for tool in tools]
        else:
            tools = {'error': results[filename]}
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(tools, f, ensure_ascii=False, indent=2)

    return results
