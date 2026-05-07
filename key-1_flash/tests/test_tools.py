from __future__ import annotations

import asyncio
from pathlib import Path

from agent.tools.registry import build_default_registry


def test_file_tools_and_search(settings, tmp_path: Path):
    async def run_test():
        registry = build_default_registry(settings)
        target = tmp_path / "note.txt"

        write = await registry.run("write_file", path=str(target), content="hello world")
        read = await registry.run("read_file", path=str(target))
        edit = await registry.run("edit_file", path=str(target), old="world", new="agent")
        search = await registry.run("search_text", root=str(tmp_path), query="agent")

        assert write.ok
        assert read.output == "hello world"
        assert edit.ok
        assert str(target) in search.output

    asyncio.run(run_test())


def test_shell_and_python_tools(settings):
    async def run_test():
        registry = build_default_registry(settings)

        shell = await registry.run("shell", command='python -c "print(123)"')
        python = await registry.run("python", code="print(456)")

        assert shell.ok
        assert "123" in shell.output
        assert python.ok
        assert "456" in python.output

    asyncio.run(run_test())
