from __future__ import annotations

import asyncio

from agent.core.reflection import ReflectionLoop
from agent.core.parser import parse_model_response
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry


class FailingThenFinalModel:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages):
        self.calls += 1
        if self.calls == 1:
            return '{"thought":"try missing","action":{"tool":"missing","args":{}}}'
        return '{"thought":"reflect","final":"handled failure"}'


class EchoTool:
    name = "echo"
    description = "Echo text."

    async def run(self, **kwargs):
        return ToolResult(ok=True, output=str(kwargs.get("text", "")))


def test_reflection_retries_after_failure(settings):
    async def run_test():
        registry = ToolRegistry()
        registry.register(EchoTool())
        loop = ReflectionLoop(settings, registry)
        model = FailingThenFinalModel()

        result = await loop.run(model, [{"role": "user", "content": "test"}])

        assert result.final == "handled failure"
        assert result.attempts == 2
        assert result.tool_results[0].ok is False

    asyncio.run(run_test())


def test_parser_handles_invalid_tool_args_without_crashing():
    parsed = parse_model_response(
        '{"thought":"bad args","action":{"tool":"shell","args":["echo","hi"]}}'
    )

    assert parsed.final is not None
    assert parsed.tool is None
