"""Base abstractions for agent tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ToolResult:
    """Result returned by a tool invocation."""

    ok: bool
    output: str
    error: str | None = None


class Tool(Protocol):
    """Protocol implemented by all tools."""

    name: str
    description: str

    async def run(self, **kwargs: Any) -> ToolResult:
        """Run the tool with keyword arguments."""
