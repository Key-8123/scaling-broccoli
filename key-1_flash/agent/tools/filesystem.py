"""Filesystem tools scoped to the current working tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.base import ToolResult


class FileReadTool:
    name = "read_file"
    description = "Read a UTF-8 text file from disk. Args: path."

    async def run(self, **kwargs: Any) -> ToolResult:
        path = Path(str(kwargs["path"])).expanduser()
        try:
            return ToolResult(ok=True, output=path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - tool errors should be captured
            return ToolResult(ok=False, output="", error=str(exc))


class FileWriteTool:
    name = "write_file"
    description = "Write UTF-8 text to a file, creating parents. Args: path, content."

    async def run(self, **kwargs: Any) -> ToolResult:
        path = Path(str(kwargs["path"])).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(kwargs.get("content", "")), encoding="utf-8")
            return ToolResult(ok=True, output=f"Wrote {path}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output="", error=str(exc))


class TextSearchTool:
    name = "search_text"
    description = "Search text files under a path. Args: root, query."

    async def run(self, **kwargs: Any) -> ToolResult:
        root = Path(str(kwargs.get("root", "."))).expanduser()
        query = str(kwargs["query"])
        matches: list[str] = []
        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                if query in text:
                    matches.append(str(path))
            return ToolResult(ok=True, output="\n".join(matches))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output="", error=str(exc))


class FileEditTool:
    name = "edit_file"
    description = "Replace text in a UTF-8 file. Args: path, old, new."

    async def run(self, **kwargs: Any) -> ToolResult:
        path = Path(str(kwargs["path"])).expanduser()
        old = str(kwargs["old"])
        new = str(kwargs["new"])
        try:
            text = path.read_text(encoding="utf-8")
            if old not in text:
                return ToolResult(ok=False, output="", error="old text not found")
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            return ToolResult(ok=True, output=f"Edited {path}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output="", error=str(exc))
