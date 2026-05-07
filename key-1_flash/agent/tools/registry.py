"""Tool registry and factory."""

from __future__ import annotations

from typing import Any

from agent.config.settings import Settings
from agent.tools.base import Tool, ToolResult
from agent.tools.filesystem import FileEditTool, FileReadTool, FileWriteTool, TextSearchTool
from agent.tools.python_exec import PythonExecutionTool
from agent.tools.shell import ShellCommandTool


class ToolRegistry:
    """Registry that routes tool calls by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def descriptions(self) -> str:
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools.values())

    async def run(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, output="", error=f"unknown tool: {name}")
        return await tool.run(**kwargs)


def build_default_registry(settings: Settings) -> ToolRegistry:
    """Build the default plugin-ready tool registry."""

    registry = ToolRegistry()
    for tool in (
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        TextSearchTool(),
        ShellCommandTool(settings),
        PythonExecutionTool(settings),
    ):
        registry.register(tool)
    return registry
