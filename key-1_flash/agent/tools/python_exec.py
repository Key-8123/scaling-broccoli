"""Python code execution tool."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from agent.config.settings import Settings
from agent.tools.base import ToolResult


class PythonExecutionTool:
    """Execute Python snippets in a subprocess."""

    name = "python"
    description = "Execute Python code in a temporary file. Args: code."

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run(self, **kwargs: Any) -> ToolResult:
        code = str(kwargs["code"])
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(code, encoding="utf-8")
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.settings.shell_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(ok=False, output="", error="python execution timed out")
        return ToolResult(
            ok=proc.returncode == 0,
            output=stdout.decode(errors="replace"),
            error=stderr.decode(errors="replace") or None,
        )
