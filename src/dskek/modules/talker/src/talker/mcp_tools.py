"""Expose tools of external MCP servers to the LLM (stdio or streamable-HTTP).

All connections live in one dedicated task: the MCP SDK uses anyio task groups, which must be
entered and exited from the same asyncio task.
"""

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import override

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from .config import ConfigError, McpServerConfig
from .tools import JsonSchema, Tool, ToolContext, ToolError, ToolRegistry, ToolSpec

logger = logging.getLogger("dskek.voice_chat.mcp")

_MAX_NAME = 64


def _tool_name(server: str, tool: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp_{server}_{tool}")[:_MAX_NAME]


class McpTool(Tool):
    def __init__(self, *, server: McpServerConfig, name: str, description: str, schema: JsonSchema, session: ClientSession) -> None:
        self._remote_name = name
        self._session = session
        self._spec = ToolSpec(
            name=_tool_name(server.name, name),
            description=f"{description or name} (via MCP server '{server.name}')",
            parameters=schema,
            risk=server.risk,
        )

    @property
    @override
    def spec(self) -> ToolSpec:
        return self._spec

    @override
    async def invoke(self, ctx: ToolContext, arguments: str) -> str:
        try:
            data = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolError(f"arguments are not valid JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ToolError("arguments must be a JSON object")
        result = await self._session.call_tool(self._remote_name, data)
        if not isinstance(result, CallToolResult):
            raise ToolError("the MCP server asked for extra input, which is not supported")
        text = "\n".join(
            block.text if isinstance(block, TextContent) else f"[{block.type} content omitted]" for block in result.content
        )
        if result.is_error:
            raise ToolError(text or "the MCP tool reported an error")
        return text or "Done."


class McpToolbox:
    """Connects to the configured servers and registers their tools in a :class:`ToolRegistry`."""

    def __init__(self, servers: Sequence[McpServerConfig], registry: ToolRegistry) -> None:
        self._servers = servers
        self._registry = registry
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._ready: asyncio.Future[None] | None = None

    async def start(self) -> None:
        if not self._servers or self._task is not None:
            return
        self._ready = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._run(), name="voice-chat-mcp")
        await self._ready

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            except Exception:
                logger.exception("error while closing MCP connections")
            self._task = None
        self._registry.unregister_prefix("mcp_")

    async def _run(self) -> None:
        assert self._ready is not None
        try:
            async with AsyncExitStack() as stack:
                for server in self._servers:
                    try:
                        count = await self._connect(stack, server)
                        logger.info("MCP server %r: %d tool(s)", server.name, count)
                    except Exception:
                        logger.exception("MCP server %r could not be started; skipping", server.name)
                self._ready.set_result(None)
                await self._stop.wait()
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                logger.exception("MCP task ended with an error")

    async def _connect(self, stack: AsyncExitStack, server: McpServerConfig) -> int:
        if server.command:
            params = StdioServerParameters(command=server.command, args=list(server.args), env=dict(server.env) or None)
            streams = await stack.enter_async_context(stdio_client(params))
        elif server.url:
            streams = await stack.enter_async_context(streamable_http_client(server.url))
        else:
            raise ConfigError(f"MCP server {server.name!r} needs either 'command' or 'url'")
        session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
        await session.initialize()

        count = 0
        for remote in (await session.list_tools()).tools:
            if server.allowed_tools and remote.name not in server.allowed_tools:
                continue
            self._registry.register(
                McpTool(
                    server=server,
                    name=remote.name,
                    description=remote.description or "",
                    schema=dict(remote.input_schema),
                    session=session,
                )
            )
            count += 1
        return count
