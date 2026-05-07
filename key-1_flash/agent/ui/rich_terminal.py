"""Rich-powered terminal interface."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from agent.config.settings import Settings


class RichTerminalUI:
    """Small terminal UI wrapper to keep presentation out of the agent core."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def banner(self, settings: Settings) -> None:
        """Render startup information."""

        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Model", settings.model_id)
        table.add_row("CPU fallback", settings.cpu_model_id)
        table.add_row("Memory", str(settings.memory_dir))
        table.add_row("Tools", "enabled" if settings.enable_tools else "disabled")
        table.add_row("Default mode", settings.default_mode)
        table.add_row("Default effort", settings.default_effort)
        self.console.print(
            Panel(table, title="[bold]Local AI Agent[/bold]", border_style="cyan")
        )

    def status(self, message: str, style: str = "cyan") -> None:
        """Render a concise status message."""

        self.console.print(f"[{style}]{message}[/{style}]")

    def assistant(self, text: str) -> None:
        """Render assistant output."""

        self.console.print(Panel(Markdown(text), title="Assistant", border_style="green"))

    def user_prompt(self) -> str:
        """Prompt for user input."""

        return Prompt.ask("[bold cyan]You[/bold cyan]")

    def memory_status(self, short_count: int, long_count: int) -> None:
        """Render memory counts."""

        self.console.print(
            f"[dim]Memory: short-term={short_count} long-term={long_count}[/dim]"
        )

    def tool_status(self, name: str, ok: bool) -> None:
        """Render tool execution status."""

        style = "green" if ok else "red"
        self.console.print(f"[{style}]tool:{name} {'ok' if ok else 'failed'}[/{style}]")

    @contextmanager
    def spinner(self, message: str) -> Iterator[None]:
        """Show a spinner around blocking work."""

        with self.console.status(message, spinner="dots"):
            yield
