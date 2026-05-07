from __future__ import annotations

import asyncio

from agent.core.agent import LocalAIAgent
from agent.memory.manager import MemoryManager
from agent.tools.registry import build_default_registry


class FakeModel:
    async def load(self):
        return None

    async def generate(self, messages):
        return '{"thought":"chat","final":"hello from local agent"}'


class ToolCallingFakeModel:
    def __init__(self) -> None:
        self.calls = 0

    async def load(self):
        return None

    async def generate(self, messages):
        self.calls += 1
        if self.calls == 1:
            return '{"thought":"need shell","action":{"tool":"shell","args":{"command":"python -c \\"print(99)\\""}}}'
        return '{"thought":"observed","final":"command returned 99"}'


class SilentUI:
    def spinner(self, message):
        class Context:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return Context()

    def status(self, *args, **kwargs):
        return None

    def tool_status(self, *args, **kwargs):
        return None

    def memory_status(self, *args, **kwargs):
        return None


def test_agent_handles_chat_message(settings):
    async def run_test():
        memory = MemoryManager(settings)
        await memory.initialize()
        agent = LocalAIAgent(
            settings,
            model=FakeModel(),
            memory=memory,
            tools=build_default_registry(settings),
            ui=SilentUI(),
        )

        response = await agent.handle_user_message("hello")

        assert response == "hello from local agent"
        assert len(memory.short_term) == 2

    asyncio.run(run_test())


def test_agent_build_mode_uses_tool_loop(settings):
    async def run_test():
        memory = MemoryManager(settings)
        await memory.initialize()
        model = ToolCallingFakeModel()
        agent = LocalAIAgent(
            settings,
            model=model,
            memory=memory,
            tools=build_default_registry(settings),
            ui=SilentUI(),
        )

        response = await agent.handle_user_message("run a python command")

        assert response == "command returned 99"
        assert model.calls == 2

    asyncio.run(run_test())


def test_agent_plan_mode_blocks_tool_loop(settings):
    async def run_test():
        memory = MemoryManager(settings)
        await memory.initialize()
        model = ToolCallingFakeModel()
        agent = LocalAIAgent(
            settings,
            model=model,
            memory=memory,
            tools=build_default_registry(settings),
            ui=SilentUI(),
        )
        agent.runtime.set_mode("plan")

        response = await agent.handle_user_message("run a python command")

        assert "did not execute" in response
        assert model.calls == 1

    asyncio.run(run_test())


def test_agent_runtime_commands(settings):
    memory = MemoryManager(settings)
    agent = LocalAIAgent(
        settings,
        model=FakeModel(),
        memory=memory,
        tools=build_default_registry(settings),
        ui=SilentUI(),
    )

    assert agent._handle_command("/mode plan") is True
    assert agent.runtime.mode == "plan"
    assert agent._handle_command("/effort high") is True
    assert agent.runtime.effort == "high"
