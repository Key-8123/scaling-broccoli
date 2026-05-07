"""Shell command execution tool."""

from __future__ import annotations

import asyncio
from typing import Any

from agent.config.settings import Settings
from agent.tools.base import ToolResult


class ShellCommandTool:
    """Run shell commands with timeout and captured output."""

    name = "shell"
    description = "Execute a terminal command. Args: command."

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run(self, **kwargs: Any) -> ToolResult:
        command = str(kwargs["command"])
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.settings.shell_timeout
            )
            output = stdout.decode(errors="replace")
            error = stderr.decode(errors="replace") or None
            return ToolResult(ok=proc.returncode == 0, output=output, error=error)
        except asyncio.TimeoutError:
            return ToolResult(ok=False, output="", error="command timed out")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output="", error=str(exc))
