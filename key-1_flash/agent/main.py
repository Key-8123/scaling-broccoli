"""Application bootstrap."""

from __future__ import annotations

import asyncio

from agent.config.settings import Settings
from agent.core.agent import LocalAIAgent
from agent.core.logging import configure_logging
from agent.memory.manager import MemoryManager
from agent.models.hf_local import LocalModel
from agent.tools.registry import build_default_registry
from agent.ui.rich_terminal import RichTerminalUI


async def async_run() -> None:
    """Start the terminal agent."""

    settings = Settings.load()
    configure_logging(settings)
    ui = RichTerminalUI()
    ui.banner(settings)

    memory = MemoryManager(settings)
    await memory.initialize()
    tools = build_default_registry(settings)
    model = LocalModel(settings, ui=ui)
    agent = LocalAIAgent(settings, model=model, memory=memory, tools=tools, ui=ui)
    await agent.initialize()
    await agent.chat_loop()


def run() -> None:
    """Synchronous console entrypoint."""

    try:
        asyncio.run(async_run())
    except KeyboardInterrupt:
        RichTerminalUI().status("Session interrupted.", style="yellow")
