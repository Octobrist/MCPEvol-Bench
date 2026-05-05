import asyncio
import logging
import pty
import threading
import subprocess
import shlex
import atexit
import time
from typing import List, Optional, Dict, Any, Tuple
from mcp.client.streamable_http import streamablehttp_client
import mcp

import asyncio
import logging
import os
import re
from contextlib import AsyncExitStack

from cachetools import LRUCache
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, FileUrl, RootModel
from pydantic.networks import AnyUrl, UrlConstraints
from mcp.types import Tool

# 可选：配置日志
logger = logging.getLogger(__name__)

# 用于跟踪当前活动的 MCP 子进程
# _CURRENT_PROC: Optional[subprocess.Popen] = None
_CURRENT_MCP_PROCESS = None
ONE_MCP_CONFIG_PATH='/root/.config/1mcp' # default
# MCP_SERVER_URL = "http://localhost:3050/mcp"  # 根据日志确定端口为 3050

class McpServerInfo(BaseModel):
    """
    Model to represent the information of an MCP server.
    """

    server_name: str
    version: str
    model_config = ConfigDict(extra="allow")
    tools: list[Tool]

class LRUCacheWithCallback(LRUCache):
    def __init__(self, maxsize, on_evict=None, *args, **kwargs):
        super().__init__(maxsize, *args, **kwargs)
        self.on_evict = on_evict

    def popitem(self):
        key, value = super().popitem()
        if self.on_evict:
            self.on_evict(key, value)
        return key, value

class MCPClient:
    def __init__(self, timeout: int = 30, max_sessions=30):
        # Initialize session and client objects
        self.timeout = timeout
        self.max_sessions = max_sessions

        def on_eviction(server_id, session):
            logger.info(f"[LRU] Evicting {server_id}")
            asyncio.create_task(self.cleanup_server(server_id))

        self.sessions: LRUCacheWithCallback[str, ClientSession] = LRUCacheWithCallback(
            max_sessions, on_evict=on_eviction
        )
        # for avoid error
        self.task: dict[str, asyncio.Task] = {}
        self.stop_event: dict[str, asyncio.Event] = {}

    async def tool_execute(self, server_id, tool_name, tool_params):
        if server_id not in self.sessions:
            raise ValueError(f"Server {server_id} is not connected.")
        session = self.sessions[server_id]
        try:
            result = await session.call_tool(tool_name, tool_params)
            return result
        except Exception as e:
            logger.error(
                f"Error executing tool {tool_name} with {tool_params} on server {server_id}: {e}"
            )
            raise ValueError(f"Error executing tool {tool_name}.")

    async def config_connect(self, config: dict, prefix: str = None):
        # Connect to an MCP server using a config file
        config = config["mcpServers"]
        for server in config:
            server_id = f"{prefix}{server}" if prefix else server
            if server_id in self.sessions:
                continue
            ready_event = asyncio.Event()

            # This is necessary to ensure in the same event loop
            async def mcp_session_runner() -> None:
                command = config[server].get("command")
                url = config[server].get("url")
                exit_stack = AsyncExitStack()
                try:
                    if command:
                        args = config[server].get("args", [])
                        env = config[server].get("env", None)
                        if env:
                            env = self._process_env_vars(env)
                        PROXY_ENV_LIST = [
                            "HTTP_PROXY",
                            "HTTPS_PROXY",
                            "NO_PROXY",
                            "http_proxy",
                            "https_proxy",
                            "no_proxy",
                        ]
                        for proxy_env in PROXY_ENV_LIST:
                            if proxy_env in os.environ:
                                env = env or {}
                                env[proxy_env] = os.environ[proxy_env]
                        await self.connect_to_server(
                            server_id, command, args, env, exit_stack
                        )
                    elif url:
                        header = config[server].get("header", None)
                        url = self._process_url_vars(url)
                        await self.connect_to_server_sse(
                            server_id, url, header, exit_stack
                        )
                    else:
                        raise ValueError(
                            "Config file must contain either a command or a url for each server"
                        )
                except Exception as e:
                    logger.error(f"Failed to connect to server {server_id}: {e}")
                ready_event.set()
                try:
                    stop_event = asyncio.Event()
                    current_task = asyncio.current_task()
                    self.stop_event[server_id] = stop_event
                    self.task[server_id] = current_task
                    assert current_task is not None, "Current task should not be None"
                    await stop_event.wait()
                finally:
                    try:
                        await exit_stack.aclose()
                    except Exception as e:
                        logger.exception("Error during exit stack close", exc_info=e)
                    logger.info(f"MCP session {server} closed")

            asyncio.create_task(mcp_session_runner())
            await ready_event.wait()

    def _process_env_vars(self, env: dict) -> dict:
        """Process environment variables in config"""
        processed_env = {}
        for key in env:
            match = re.findall(r"\${(.*)}", env[key])
            processed_value = env[key]
            for m in match:
                if m in os.environ:
                    processed_value = processed_value.replace(
                        f"${{{m}}}", os.environ[m]
                    )
                else:
                    raise ValueError(
                        f"Environment variable {m} not found for env: {env}"
                    )
            processed_env[key] = processed_value
        return processed_env

    def _process_url_vars(self, url: str) -> str:
        """Process environment variables in URL"""
        match = re.findall(r"\${(.*)}", url)
        processed_url = url
        for m in match:
            if m in os.environ:
                processed_url = processed_url.replace(f"${{{m}}}", os.environ[m])
            else:
                raise ValueError(f"Environment variable {m} not found for URL: {url}")
        return processed_url

    async def connect_to_server_sse(
        self, server_id: str, url: str, header=None, exit_stack: AsyncExitStack = None
    ):
        # Connect to the server using SSE
        try:
            sse_transport = await exit_stack.enter_async_context(
                sse_client(url, header)
            )
            sse, write = sse_transport
            session = await exit_stack.enter_async_context(
                ClientSession(sse, write, self.timeout)
            )
            await asyncio.wait_for(session.initialize(), timeout=self.timeout)
            self.sessions[server_id] = session
            logger.info(f"Connected to server {server_id}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to SSE server {server_id}")
            raise
        except Exception as e:
            logger.error(f"Error connecting to SSE server {server_id}: {e}")
            raise

    async def connect_to_server(
        self,
        server_id: str,
        command: str,
        args: list,
        env: dict | None = None,
        exit_stack: AsyncExitStack = None,
    ):
        # Connect to an MCP server
        try:
            server_params = StdioServerParameters(command=command, args=args, env=env)
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await exit_stack.enter_async_context(ClientSession(stdio, write))
            await asyncio.wait_for(session.initialize(), timeout=self.timeout)
            self.sessions[server_id] = session
            logger.info(f"Connected to server {server_id}.")
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to server {server_id}")
            # await self.cleanup_server(server_id)
            raise
        except Exception as e:
            logger.error(f"Error connecting to server {server_id}: {e}")
            # await self.cleanup_server(server_id)
            raise

    async def collect_server_info(self, server_id: str):
        # Collect information from a single server with error handling
        try:
            session: ClientSession = self.sessions.get(server_id)
            if not session:
                logger.error(f"No session found for server {server_id}")
                return None

            response = await asyncio.wait_for(
                session.list_tools(), timeout=self.timeout
            )
            mcp_version = session._client_info.version
            model_config = session._client_info.model_config
            info = McpServerInfo(
                server_name=server_id,
                version=mcp_version,
                model_config=model_config,
                tools=response.tools,
            )
            return info.model_dump()
        except asyncio.TimeoutError:
            logger.error(f"Timeout collecting info from server {server_id}")
            return None
        except Exception as e:
            logger.error(f"Error collecting info from server {server_id}: {e}")
            return None

    async def collect_all_info(self):
        # Collect all information from all servers
        all_info = {}
        tasks = []

        for server_id in self.sessions:
            tasks.append(self.collect_server_info(server_id))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, (server_id, result) in enumerate(zip(self.sessions.keys(), results)):
                if isinstance(result, Exception):
                    logger.error(f"Exception for server {server_id}: {result}")
                elif result is not None:
                    all_info[server_id] = result

        return all_info

    async def list_tools(self, server_id: str) -> dict[str, dict]:
        """Lists all available tools from a connected MCP server."""
        if server_id not in self.sessions:
            logger.warning(f"Server {server_id} not connected, cannot list tools.")
            return {}
        session = self.sessions[server_id]
        try:
            logger.info(f"Listing tools for server {server_id}")
            list_tools = await session.list_tools()
            list_tools = list_tools.tools
            logger.info(f"Tools for {server_id}: {list(list_tools)}")
            actual_tools_dict = {x.name: x for x in list_tools}
            return actual_tools_dict
        except Exception as e:
            logger.error(f"Error listing tools for {server_id}: {e}", exc_info=True)
            return {}

    async def cleanup_server(self, server_id: str):
        ev = self.stop_event.get(server_id)
        t = self.task.get(server_id)

        if ev is not None:
            ev.set()

        if t is not None:
            await asyncio.shield(t)

        self.sessions.pop(server_id, None)
        self.stop_event.pop(server_id, None)
        self.task.pop(server_id, None)

    async def cleanup(self):
        server_ids = list(self.sessions.keys())
        for server_id in server_ids:
            try:
                await asyncio.shield(self.cleanup_server(server_id))
            except Exception as e:
                logger.error(f"cleanup_server failed for {server_id}: {e}")

def start_mcp_server(timeout_on_start: int = 10) -> bool:
    """
    启动一个已通过 1mcp 注册的 MCP 服务器（临时使用，用完应调用 stop_mcp_server）。

    特点：
      - 启动后立即返回（不阻塞）
      - 自动在程序退出时清理子进程
      - 可配合后续的工具调用（stdio 或 HTTP）

    返回:
        bool: True 表示启动命令已成功派发（不保证服务完全就绪）
    """
    global _CURRENT_MCP_PROCESS

    # 如果已有运行中的 MCP server，先停止
    if _CURRENT_MCP_PROCESS is not None:
        stop_mcp_server()
    time.sleep(2)
    # 获取命令
    try:
        proc = subprocess.Popen(
            ['1mcp'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # 行缓冲
        )
        _CURRENT_MCP_PROCESS = proc
        # 注册退出清理（确保即使程序崩溃也能 kill）
        atexit.register(stop_mcp_server)
        time.sleep(10)
        # 简单等待（可选）：确保进程没立即崩溃
        # 注意：stdio 服务通常无 "ready" 信号，所以这里不强等
        return True and _CURRENT_MCP_PROCESS is not None
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False

_CURRENT_OUTPUT = ""  # 累积 stdout + stderr 的最新内容（可定期清空）

def _read_output(pipe):
    global _CURRENT_OUTPUT
    for line in iter(pipe.readline, ''):
        _CURRENT_OUTPUT += line  # 追加新输出
    pipe.close()


def start_mcp_server(timeout=60) -> bool:
    global _CURRENT_MCP_PROCESS, _CURRENT_OUTPUT
    if _CURRENT_MCP_PROCESS:
        stop_mcp_server()
    try:
        proc = subprocess.Popen(
            ['1mcp'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
            text=True,
            bufsize=1
        )
        _CURRENT_MCP_PROCESS = proc
        _CURRENT_OUTPUT = ""  # 清空旧输出
        # 启动后台线程读取输出
        threading.Thread(target=_read_output, args=(proc.stdout,), daemon=True).start()
        atexit.register(stop_mcp_server)
        return True
    except Exception as e:
        print(f"启动失败: {e}")
        return False


def get_mcp_output(clear: bool = False) -> str:
    """
    获取当前累积的输出内容。
    如果 clear=True，则获取后清空缓存。
    """
    global _CURRENT_OUTPUT
    output = _CURRENT_OUTPUT
    if clear:
        _CURRENT_OUTPUT = ""
    return output

def stop_mcp_server():
    """停止当前 MCP 服务器"""
    global _CURRENT_MCP_PROCESS
    if _CURRENT_MCP_PROCESS:
        try:
            _CURRENT_MCP_PROCESS.terminate()
            _CURRENT_MCP_PROCESS.wait(timeout=5)
        except:
            pass
        _CURRENT_MCP_PROCESS = None

def run_1mcp_command(command: str, timeout=60) -> Dict[str, Any]:
    """
    执行一条 1mcp 命令（如 add 或 remove），并返回执行结果。

    参数:
        command (str): 完整的命令字符串，例如:
            "1mcp mcp add myserver -- npx -y mypkg@1.0.0"
            "1mcp mcp remove myserver"

    返回:
        dict: 包含执行结果的字典，字段包括:
            - success (bool): 是否成功
            - stdout (str): 标准输出
            - stderr (str): 标准错误
            - returncode (int): 进程返回码
            - command (str): 执行的命令（用于日志）
    """
    try:
        # 使用 shlex.split 安全地拆分命令（支持带空格的路径等）
        cmd_list = shlex.split(command)

        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=timeout  # 防止卡死（可根据需要调整）
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "command": command
        }

    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "命令未找到: '1mcp' 未安装或不在 PATH 中",
            "returncode": -1,
            "command": command
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "命令执行超时",
            "returncode": -2,
            "command": command
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"执行异常: {str(e)}",
            "returncode": -3,
            "command": command
        }

async def fetch_mcp_tools_safe(
        mcp_server_url: str,
        token: Optional[str] = None,
        timeout: int = 30,
        raise_on_error: bool = False
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    安全地获取 MCP 工具列表，带完整容错。

    参数:
        mcp_server_url (str): MCP 服务地址，如 "http://localhost:3050/mcp"
        token (str, optional): 认证令牌
        timeout (int): 超时时间（秒）
        raise_on_error (bool): 若为 True，错误时抛异常；否则返回空列表+错误信息

    返回:
        tuple: (tools_list, error_message)
            - tools_list: 成功时为工具列表，失败时为 []
            - error_message: 成功时为 None，失败时为错误描述
    """
    try:
        # 使用 asyncio.wait_for 实现超时
        async with asyncio.timeout(timeout):
            async with streamablehttp_client(mcp_server_url) as (read_stream, write_stream, _):
                async with mcp.ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    # tools_info = [tool.name for tool in tools_result.tools]
                    tools_info = []
                    for tool in tools_result.tools:
                        tools_info.append({
                            "name": tool.name,
                            "description": tool.description or "",
                            "inputSchema": tool.inputSchema or {}
                        })
                    return tools_info, None

    except asyncio.TimeoutError:
        error_msg = f"❌ 连接超时（>{timeout}秒）: {mcp_server_url}"
        logger.error(error_msg)
        if raise_on_error:
            raise
        return [], error_msg

    except (ConnectionError, OSError) as e:
        error_msg = f"❌ 无法连接到 MCP 服务器: {mcp_server_url} | 原因: {str(e)}"
        logger.error(error_msg)
        if raise_on_error:
            raise
        return [], error_msg
    except mcp.McpError as e:  # ✅ 关键修复：使用 McpError
        # 例如: McpError(-32601, "Method not found") 表示不支持 list_tools
        error_msg = f"❌ MCP 协议错误: {str(e)}"
        logger.error(error_msg)
        if raise_on_error:
            raise
        return [], error_msg

    except Exception as e:
        error_msg = f"❌ 未知错误: {type(e).__name__}: {str(e)}"
        logger.exception("获取 MCP 工具时发生未预期异常")  # 记录完整 traceback
        if raise_on_error:
            raise
        return [], error_msg

async def run_1mcp_tool(tool_name, arguments) -> None:
    async with streamablehttp_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
        async with mcp.ClientSession(read_stream, write_stream) as session:
            # 初始化会话
            await session.initialize()
            # print(f"正在调用 {tool_name} 工具...")
            mcp_tool_name = 'custom_1mcp_'+tool_name.split('_1mcp_')[-1]
            result = await session.call_tool(
                name=mcp_tool_name,
                arguments=arguments
            )
            # print(result)
            # raw_text = extract_text_content(result.content)
            # try:
            #     parsed_data = json.loads(raw_text)
            #     save_to_file(parsed_data)  # 传入结构化数据
            # except json.JSONDecodeError:
            #     save_to_file(raw_text)     # 传入原始字符串
    return result

async def get_1mcp_status(mcp_server_url):
    async with streamablehttp_client(mcp_server_url) as (read_stream, write_stream, _):
        async with mcp.ClientSession(read_stream, write_stream) as session:
            # 初始化 MCP 会话
            await session.initialize()
            # 获取可用工具列表
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]
            # print(f"Available tools: {', '.join(tool_names)}")
            print(f"Tools Number {len(tool_names)}")
    return tools_result, tool_names
