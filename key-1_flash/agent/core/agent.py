"""High-level terminal agent orchestration."""

from __future__ import annotations

import logging

from agent.config.settings import Settings
from agent.core.parser import parse_model_response
from agent.core.prompts import build_system_prompt
from agent.core.reflection import ReflectionLoop
from agent.core.runtime import AgentRuntimeState
from agent.memory.manager import MemoryManager
from agent.tools.registry import ToolRegistry
from agent.ui.rich_terminal import RichTerminalUI

LOGGER = logging.getLogger(__name__)


class LocalAIAgent:
    """Terminal-first local AI assistant."""

    def __init__(
        self,
        settings: Settings,
        model: object,
        memory: MemoryManager,
        tools: ToolRegistry,
        ui: RichTerminalUI,
    ) -> None:
        self.settings = settings
        self.model = model
        self.memory = memory
        self.tools = tools
        self.ui = ui
        self.reflection = ReflectionLoop(settings, tools)
        self.runtime = AgentRuntimeState()
        try:
            self.runtime.set_mode(settings.default_mode)
            self.runtime.set_effort(settings.default_effort)
        except ValueError as exc:
            LOGGER.warning("Ignoring invalid runtime default: %s", exc)

    async def initialize(self) -> None:
        """Initialize heavyweight components."""

        with self.ui.spinner("Loading local model. First run may download weights..."):
            await self.model.load()  # type: ignore[attr-defined]
        active_model = getattr(self.model, "model_id", self.settings.model_id)
        self.ui.status(f"Model ready: {active_model}", style="green")

    async def chat_loop(self) -> None:
        """Run the interactive terminal chat loop."""

        self.ui.status(
            "Type /exit to quit, /memory, /mode plan|build, /effort low|medium|high.",
            style="cyan",
        )
        while True:
            user_text = self.ui.user_prompt().strip()
            if user_text in {"/exit", "exit", "quit"}:
                self.ui.status("Goodbye.", style="cyan")
                return
            if user_text.startswith("/"):
                handled = self._handle_command(user_text)
                if not handled:
                    self.ui.status(
                        "Unknown command. Try /mode, /effort, /memory, or /help.",
                        style="yellow",
                    )
                continue
            if not user_text:
                continue
            final = await self.handle_user_message(user_text)
            self.ui.assistant(final)

    async def handle_user_message(self, user_text: str) -> str:
        """Process a single user message."""

        self.memory.add("user", user_text)
        context = self.memory.relevant_context(user_text)
        profile = self.runtime.effort_profile
        use_tools = self.settings.enable_tools and self.runtime.mode == "build"
        system = build_system_prompt(
            self.tools,
            use_tools,
            mode=self.runtime.mode,
            effort=self.runtime.effort,
            max_attempts=profile.max_attempts,
        )
        if context:
            system += "\nRelevant memory:\n" + context

        messages = [{"role": "system", "content": system}]
        messages.extend(
            {"role": item.role, "content": item.content}
            for item in self.memory.short_term
            if item.role in {"user", "assistant"}
        )

        with self.ui.spinner("Thinking..."):
            if self.runtime.mode == "plan":
                final = await self._plan_chat(messages)
            elif use_tools:
                result = await self.reflection.run(
                    self.model,
                    messages,
                    max_attempts=profile.max_attempts,
                )
                for index, tool_result in enumerate(result.tool_results, start=1):
                    self.ui.tool_status(f"step-{index}", tool_result.ok)
                final = result.final
                LOGGER.info(
                    "Completed response in %s attempts with %s effort",
                    result.attempts,
                    self.runtime.effort,
                )
            else:
                final = await self._direct_chat(messages)
        self.memory.add("assistant", final)
        return final

    def _handle_command(self, command: str) -> bool:
        """Handle terminal slash commands."""

        parts = command.split()
        name = parts[0].lower()
        value = parts[1].lower() if len(parts) > 1 else None

        if name == "/memory":
            self.ui.memory_status(
                len(self.memory.short_term), len(self.memory.load_long_term())
            )
            return True
        if name == "/mode":
            if value is not None:
                try:
                    self.runtime.set_mode(value)
                except ValueError as exc:
                    self.ui.status(str(exc), style="red")
                    return True
            self.ui.status(self.runtime.mode_summary(), style="cyan")
            return True
        if name == "/effort":
            if value is not None:
                try:
                    self.runtime.set_effort(value)
                except ValueError as exc:
                    self.ui.status(str(exc), style="red")
                    return True
            self.ui.status(self.runtime.effort_summary(), style="cyan")
            return True
        if name == "/help":
            self.ui.status(
                "/mode plan|build  /effort low|medium|high  /memory  /exit",
                style="cyan",
            )
            return True
        return False

    async def _plan_chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a planning-only response with tools explicitly blocked."""

        chat_messages = list(messages)
        chat_messages[0] = {
            "role": "system",
            "content": (
                "You are in PLAN MODE. You may only think, reason, decompose the "
                "task, identify risks, and propose steps. Do not run commands, call "
                "tools, edit files, or claim that you inspected local state. Return "
                "plain text, not JSON."
            ),
        }
        try:
            raw = await self.model.generate(chat_messages)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Plan generation failed")
            return f"Model generation failed: {exc}"
        parsed = parse_model_response(raw)
        if parsed.final:
            return parsed.final
        if parsed.tool:
            return (
                f"Plan mode is active, so I did not execute `{parsed.tool}`. "
                "Switch to `/mode build` when you want me to perform the work."
            )
        return raw.strip()

    async def _direct_chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a normal conversational response without tool pressure."""

        chat_messages = list(messages)
        chat_messages[0] = {
            "role": "system",
            "content": (
                "You are a helpful local terminal assistant. Reply naturally and "
                "concisely. Do not call tools. Do not return JSON."
            ),
        }
        try:
            raw = await self.model.generate(chat_messages)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Direct chat generation failed")
            return f"Model generation failed: {exc}"
        parsed = parse_model_response(raw)
        return parsed.final if parsed.final else raw.strip()
