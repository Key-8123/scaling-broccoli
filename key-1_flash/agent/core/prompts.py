"""Prompt construction for the local agent."""

from __future__ import annotations

from agent.tools.registry import ToolRegistry


SYSTEM_PROMPT = """You are a local autonomous AI assistant running in a terminal.
You can chat normally, plan tasks, and call tools when useful.

When you need a tool, respond with exactly one JSON object:
{"thought": "...", "action": {"tool": "tool_name", "args": {...}}}

When no tool is needed, respond with:
{"thought": "...", "final": "..."}

Keep tool arguments explicit. Do not invent files or command results."""


def build_system_prompt(
    tools: ToolRegistry,
    enable_tools: bool,
    mode: str = "build",
    effort: str = "medium",
    max_attempts: int | None = None,
) -> str:
    """Build the runtime system prompt."""

    autonomy = (
        f"\nRuntime mode: {mode}.\nReasoning effort: {effort}"
        + (f" ({max_attempts} max attempts)." if max_attempts else ".")
    )
    if mode == "plan":
        return (
            SYSTEM_PROMPT
            + autonomy
            + "\nPLAN MODE: Do not call tools, modify files, run shell commands, "
            "or execute Python. Think through the task and return a concrete plan."
            "\nTools are disabled."
        )
    if not enable_tools:
        return SYSTEM_PROMPT + autonomy + "\nTools are disabled."
    return SYSTEM_PROMPT + autonomy + "\nAvailable tools:\n" + tools.descriptions()
